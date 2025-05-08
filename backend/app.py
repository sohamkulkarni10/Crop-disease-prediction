from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
import numpy as np

from PIL import Image
import logging
import cloudinary
import cloudinary.uploader

from datetime import datetime
import tensorflow as tf
import gdown
import json
import requests
from flask_cors import CORS
import google.generativeai as genai

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    MetaData,
)
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager  # For logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import Integer
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import os
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from langchain_community.chat_message_histories import ChatMessageHistory
import atexit


class ContextItem(BaseModel):
    query: str
    response: str


class Item(BaseModel):
    query: str | None = None
    context: list[ContextItem] | None = None


os.environ["GOOGLE_API_KEY"] = "AIzaSyBCmTwvxa2-Vt7pMlRVgqX19bHj5628t3w"

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


DATABASE_URL = "sqlite:///chat_history.db"
Base = declarative_base()


app = Flask(__name__)
CORS(app)

DATABASE_URL = "sqlite:///./crop_data.db"


class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False)
    messages = relationship("Message", back_populates="session")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    session = relationship("Session", back_populates="messages")


engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_message(session_id: str, role: str, content: str):
    db = next(get_db())
    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if not session:
            session = Session(session_id=session_id)
            db.add(session)
            db.commit()
            db.refresh(session)

        db.add(Message(session_id=session.id, role=role, content=content))
        db.commit()
    except SQLAlchemyError:
        db.rollback()
    finally:
        db.close()


def load_session_history(session_id: str) -> BaseChatMessageHistory:
    db = next(get_db())
    chat_history = ChatMessageHistory()
    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if session:
            for message in session.messages:
                chat_history.add_message(
                    {"role": message.role, "content": message.content}
                )
    except SQLAlchemyError:
        pass
    finally:
        db.close()

    return chat_history


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = load_session_history(session_id)
    return store[session_id]


def save_all_sessions():
    for session_id, chat_history in store.items():
        for message in chat_history.messages:
            save_message(session_id, message["role"], message["content"])


class Query(BaseModel):
    query: str | None = None
    session_id: str | None = "default"


@app.route("/final", methods=["POST"])
def final():
    data = request.get_json()
    print(data)
    if not data:
        return jsonify({"error": "Missing JSON payload"}), 400

    query = data.get("query")
    session_id = data.get("session_id")

    if not query or not session_id:
        return jsonify({"error": "Missing 'query' or 'session_id' in request"}), 400

    logger.info(f"Received query: '{query}' for session_id: '{session_id}'")

    chat_system_prompt = """
You are Smart Assistant, an AI assistant focused on general agricultural queries in India. Provide clear, practical advice to farmers based on the conversation history.

Analyze the farmer's latest question carefully using the chat history. Follow the rules below strictly:

<guidelines>
1. Use conversation history for context.
2. If you don't have enough information from the history to answer, say so clearly. You may add general agricultural advice only if relevant.
3. Use simple, farmer-friendly language. Avoid jargon.
4. Prefer Indian terms, units, and examples where applicable.
5. Be concise, practical, and actionable.
6. For treatments or chemical use, always mention safety steps and recommend consulting labels or local agricultural experts.
7. Stay on topic—focus on agriculture. Politely decline unrelated questions.
8. Reply in a clear manner, using paragraphs or lists where it improves readability.
</guidelines>
"""

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", chat_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    basic_chat_chain = chat_prompt | llm

    conversational_chain = RunnableWithMessageHistory(
        basic_chat_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    try:
        save_message(session_id, "human", query)
        logger.info(f"Saved human message for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to save human message for session {session_id}: {e}")

    try:
        ai_message = conversational_chain.invoke(
            {"input": query}, config={"configurable": {"session_id": session_id}}
        )
        result = (
            ai_message.content if hasattr(ai_message, "content") else str(ai_message)
        )

        logger.info(f"Generated AI response for session {session_id}")

    except Exception as e:
        logger.error(
            f"Error invoking conversational chain for session {session_id}: {e}"
        )
        return jsonify({"error": "Failed to generate AI response"}), 500

    try:
        save_message(session_id, "ai", result)
        logger.info(f"Saved AI message for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to save AI message for session {session_id}: {e}")

    return jsonify({"response": result.replace("*", "")})


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(String, unique=True, index=True, nullable=False)
    disease_name = Column(String, nullable=False)
    confidence = Column(Float)
    image_url = Column(String)
    timestamp = Column(DateTime, default=datetime.now)


class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    disease_context = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.now)


@contextmanager
def get_db_session():
    """Provide a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database Error: {e}")
        raise
    finally:
        db.close()


cloudinary.config(
    cloud_name="dahksocqw",
    api_key="774146768391685",
    api_secret="j0T2SeocxZy_vTQ2KX-ClIygY0s",
)

GEMINI_API_KEY = "AIzaSyBCmTwvxa2-Vt7pMlRVgqX19bHj5628t3w"
genai.configure(api_key=GEMINI_API_KEY)

gemini_config = {
    "temperature": 0.7,
    "max_output_tokens": 500,
    "top_p": 0.95,
    "top_k": 40,
}

MODEL_FILENAME = "vgg16_multi_class_model_fine_tuned.h5"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)
GOOGLE_DRIVE_MODEL_ID =  "1xSdnUjcBFfY0emWcCKfxkoEqOW0i2VvR"


def download_model_from_drive():
    try:
        logger.info("Downloading model from Google Drive...")

        gdown.download(id=GOOGLE_DRIVE_MODEL_ID, output=MODEL_PATH, quiet=False)
        logger.info(f"Model downloaded successfully to {MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error downloading model from Google Drive: {e}")
        return False


model = None


def load_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            if not download_model_from_drive():
                return False
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully.")
        logger.info(f"Model input shape: {model.input_shape}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


class_labels = [
    "Apple__Apple_scab",
    "Apple_Black_rot",
    "Apple_Cedar_apple_rust",
    "Apple_healthy",
    "Blueberry_healthy",
    "Brownspot",
    "Cashew anthracnose",
    "Cashew gumosis",
    "Cashew healthy",
    "Cashew leaf miner",
    "Cashew red rust",
    "Cassava bacterial blight",
    "Cassava brown spot",
    "Cassava green mite",
    "Cassava healthy",
    "Cassava mosaic",
    "Cherry(including_sour)healthy",
    "Cherry(including_sour)Powdery_mildew",
    "Corn(maize)Cercospora_leaf_spot Gray_leaf_spot",
    "Corn(maize)Common_rust",
    "Corn_(maize)healthy",
    "Corn(maize)Northern_Leaf_Blight",
    "Cotton American Bollworm",
    "Cotton Anthracnose",
    "Cotton Aphid",
    "Cotton Bacterial Blight",
    "Cotton bollrot",
    "Cotton bollworm",
    "Cotton Healthy",
    "cotton mealy bug",
    "Cotton pink bollworm",
    "Cotton thirps",
    "cotton whitefly",
    "Grape_Black_rot",
    "Grape_Esca(Black_Measles)",
    "Grape__healthy",
    "Grape_Leaf_blight(Isariopsis_Leaf_Spot)",
    "Gray_Leaf_Spot",
    "Healthy Maize",
    "Leaf Curl",
    "Leaf smut",
    "maize ear rot",
    "Maize fall armyworm",
    "Maize grasshoper",
    "Maize healthy",
    "Maize leaf beetle",
    "Maize leaf blight",
    "Maize leaf spot",
    "maize stem borer",
    "Maize streak virus",
    "Orange__Haunglongbing(Citrus_greening)",
    "Peach__Bacterial_spot",
    "Peach_healthy",
    "Pepper,_bell_Bacterial_spot",
    "Pepper,_bell_healthy",
    "Potato_Early_blight",
    "Potato_healthy",
    "Potato_Late_blight",
    "Raspberry_healthy",
    "red cotton bug",
    "Rice Becterial Blight",
    "Rice Blast",
    "Rice Tungro",
    "Soybean_healthy",
    "Squash_Powdery_mildew",
    "Strawberry_healthy",
    "Strawberry_Leaf_scorch",
    "Sugarcane Healthy",
    "Sugarcane mosaic",
    "Sugarcane RedRot",
    "Sugarcane RedRust",
    "Tomato leaf blight",
    "Tomato leaf curl",
    "Tomato verticulium wilt",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites Two-spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Wheat aphid",
    "Wheat black rust",
    "Wheat Brown leaf rust",
    "Wheat Flag Smut",
    "Wheat Healthy",
    "Wheat leaf blight",
    "Wheat mite",
    "Wheat powdery mildew",
    "Wheat scab",
    "Wheat Stem fly",
    "Wheat__Yellow_Rust",
    "Wilt",
    "Yellow Rust Sugarcane",
]

HISTORY_FOLDER = "history"
os.makedirs(HISTORY_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("cdr7.html")  # Make


@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        if not load_model():
            return (
                jsonify(
                    {"error": "Model could not be loaded. Please try again later."}
                ),
                500,
            )

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    prediction_id = str(uuid.uuid4())
    temp_filepath = os.path.join(HISTORY_FOLDER, f"{prediction_id}.jpg")

    try:
        file.save(temp_filepath)

        image = tf.keras.preprocessing.image.load_img(
            temp_filepath, target_size=(128, 128), color_mode="rgb"
        )
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        prediction_probs = model.predict(input_arr)
        result_index = np.argmax(prediction_probs)
        result = class_labels[result_index]
        confidence = float(np.max(prediction_probs[0]))
        logger.info(f"Prediction made: {result} with confidence {confidence:.4f}")

        try:
            cloudinary_response = cloudinary.uploader.upload(
                temp_filepath, folder="crop_disease", public_id=prediction_id
            )
            image_url = cloudinary_response["secure_url"]
            logger.info(f"Image uploaded to Cloudinary: {image_url}")
        except Exception as e:
            logger.error(f"Error uploading to Cloudinary: {e}")

            image_url = None  #

        try:
            with get_db_session() as db:
                new_prediction = Prediction(
                    prediction_id=prediction_id,
                    disease_name=result,
                    confidence=confidence,
                    image_url=image_url,
                    timestamp=datetime.now(),
                )
                db.add(new_prediction)

            logger.info(f"Prediction stored in SQLite with ID: {prediction_id}")
        except Exception as e:
            logger.error(f"Error storing prediction in SQLite: {e}")
            return jsonify({"error": f"Error storing prediction data: {e}"}), 500

        return jsonify({"prediction": result, "image_url": image_url})

    except Exception as e:
        logger.error(f"Error during prediction process: {e}")
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 500
    finally:
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
                logger.info(f"Removed temporary file: {temp_filepath}")
            except OSError as rm_err:
                logger.error(f"Error removing temporary file {temp_filepath}: {rm_err}")


@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        data = request.json
        user_message = data.get("message", "")
        disease = data.get("disease", None)
        if disease:
            if user_message.startswith("Tell me about"):
                prompt = f"You are an agricultural expert specializing in crop diseases. Provide detailed information about {disease}, including its causes, symptoms, and effective remedies or treatments. Format your response in a concise, easy-to-understand manner."
            else:
                prompt = f"You are an agricultural expert specializing in crop diseases. The user previously asked about {disease} and now asks: '{user_message}'. Provide a helpful, accurate response."
        else:
            prompt = f"You are an agricultural expert specializing in crop diseases. A farmer asks: '{user_message}'. Provide a helpful, accurate response."

        try:
            gemini_model = genai.GenerativeModel(
                "gemini-pro", generation_config=gemini_config
            )
            response = gemini_model.generate_content(prompt)
            bot_response = response.text
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            bot_response = "I'm having trouble accessing my knowledge base right now. Please try again later."

        try:
            with get_db_session() as db:
                chat_entry = ChatHistory(
                    user_message=user_message,
                    bot_response=bot_response,
                    disease_context=disease,
                    timestamp=datetime.now(),
                )
                db.add(chat_entry)
            logger.info("Chat entry stored in SQLite.")
        except Exception as e:
            logger.error(f"Error storing chat history in SQLite: {e}")

        return jsonify({"response": bot_response})

    except Exception as e:
        logger.error(f"Error in chatbot endpoint: {e}")
        return (
            jsonify(
                {
                    "response": "I'm sorry, I encountered an error. Please try again later."
                }
            ),
            500,
        )


@app.route("/history")
def history():
    try:
        with get_db_session() as db:

            history_records = (
                db.query(Prediction).order_by(Prediction.timestamp.desc()).all()
            )

            history_data = [
                {
                    "prediction_id": record.prediction_id,
                    "disease_name": record.disease_name,
                    "image_url": record.image_url,
                    "timestamp": (
                        record.timestamp.isoformat() if record.timestamp else None
                    ),
                }
                for record in history_records
            ]
        return jsonify({"history": history_data})
    except Exception as e:
        logger.error(f"Error fetching history from SQLite: {e}")
        return jsonify({"error": f"Error fetching history: {e}"}), 500

@app.route('/final', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query', '')
    session_id = data.get('session_id', '')
    language = data.get('language', 'english')

    print(f"Received Query: {query}")
    print(f"Language: {language}")
    
    # Add your logic here: you could handle Marathi differently if needed
    if language == 'marathi':
        response = f"तुमचं प्रश्न: {query}"  # simple echo
    else:
        response = f"Your query was: {query}"  # simple echo for English

    return jsonify({'response': response})


def create_db_tables():
    try:
        logger.info("Creating database tables if they don't exist...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables checked/created.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")


if __name__ == "__main__":
    create_db_tables()

    if load_model():
        logger.info("Starting Flask development server...")
        app.run(debug=True, host="0.0.0.0", port=5000)
    else:
        logger.error("Failed to load model on startup. Application cannot start.")







# from flask import Flask, render_template, request, jsonify, send_from_directory
# import os
# import uuid
# import numpy as np

# from PIL import Image
# import logging
# import cloudinary
# import cloudinary.uploader

# from datetime import datetime
# import tensorflow as tf
# import gdown
# import json
# import requests
# from flask_cors import CORS
# import google.generativeai as genai


# import speech_recognition as sr
# import tempfile
# import os
# from werkzeug.utils import secure_filename



# from sqlalchemy import (
#     create_engine,
#     Column,
#     Integer,
#     String,
#     Float,
#     DateTime,
#     Text,
#     MetaData,
# )
# from sqlalchemy.orm import sessionmaker, declarative_base
# from sqlalchemy.exc import SQLAlchemyError
# from contextlib import contextmanager  # For logging.basicConfig(level=logging.INFO)

# logger = logging.getLogger(__name__)

# from langchain_google_genai import ChatGoogleGenerativeAI
# from fastapi import FastAPI
# from pydantic import BaseModel
# from langchain.chains import RetrievalQA
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from sqlalchemy import Integer
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.schema import HumanMessage, AIMessage

# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores.faiss import FAISS
# import os
# from fastapi.middleware.cors import CORSMiddleware
# from sqlalchemy.orm import sessionmaker, relationship, declarative_base
# from sqlalchemy.exc import SQLAlchemyError
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
# from langchain_community.chat_message_histories import ChatMessageHistory
# import atexit


# class ContextItem(BaseModel):
#     query: str
#     response: str


# class Item(BaseModel):
#     query: str | None = None
#     context: list[ContextItem] | None = None


# os.environ["GOOGLE_API_KEY"] = "AIzaSyBCmTwvxa2-Vt7pMlRVgqX19bHj5628t3w"

# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


# DATABASE_URL = "sqlite:///chat_history.db"
# Base = declarative_base()


# app = Flask(__name__)
# CORS(app)

# DATABASE_URL = "sqlite:///./crop_data.db"


# class Session(Base):
#     __tablename__ = "sessions"
#     id = Column(Integer, primary_key=True)
#     session_id = Column(String, unique=True, nullable=False)
#     messages = relationship("Message", back_populates="session")


# class Message(Base):
#     __tablename__ = "messages"
#     id = Column(Integer, primary_key=True)
#     session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
#     role = Column(String, nullable=False)
#     content = Column(Text, nullable=False)
#     session = relationship("Session", back_populates="messages")


# engine = create_engine(DATABASE_URL)
# Base.metadata.create_all(engine)
# SessionLocal = sessionmaker(bind=engine)


# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


# def save_message(session_id: str, role: str, content: str):
#     db = next(get_db())
#     try:
#         session = db.query(Session).filter(Session.session_id == session_id).first()
#         if not session:
#             session = Session(session_id=session_id)
#             db.add(session)
#             db.commit()
#             db.refresh(session)

#         db.add(Message(session_id=session.id, role=role, content=content))
#         db.commit()
#     except SQLAlchemyError:
#         db.rollback()
#     finally:
#         db.close()


# def load_session_history(session_id: str) -> BaseChatMessageHistory:
#     db = next(get_db())
#     chat_history = ChatMessageHistory()
#     try:
#         session = db.query(Session).filter(Session.session_id == session_id).first()
#         if session:
#             for message in session.messages:
#                 chat_history.add_message(
#                     {"role": message.role, "content": message.content}
#                 )
#     except SQLAlchemyError:
#         pass
#     finally:
#         db.close()

#     return chat_history


# store = {}


# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = load_session_history(session_id)
#     return store[session_id]


# def save_all_sessions():
#     for session_id, chat_history in store.items():
#         for message in chat_history.messages:
#             save_message(session_id, message["role"], message["content"])


# class Query(BaseModel):
#     query: str | None = None
#     session_id: str | None = "default"


# @app.route("/final", methods=["POST"])
# def final():
#     data = request.get_json()
#     print(data)
#     if not data:
#         return jsonify({"error": "Missing JSON payload"}), 400

#     query = data.get("query")
#     session_id = data.get("session_id")

#     if not query or not session_id:
#         return jsonify({"error": "Missing 'query' or 'session_id' in request"}), 400

#     logger.info(f"Received query: '{query}' for session_id: '{session_id}'")

#     # Updated system prompt to include Marathi response instruction
#     chat_system_prompt = """
# You are Smart Assistant, an AI assistant focused on general agricultural queries in India. Provide clear, practical advice to farmers based on the conversation history.

# Analyze the farmer's latest question carefully using the chat history. Follow the rules below strictly:

# <guidelines>
# 1. Use conversation history for context.
# 2. If you don't have enough information from the history to answer, say so clearly. You may add general agricultural advice only if relevant.
# 3. Use simple, farmer-friendly language. Avoid jargon.
# 4. Use Indian terms, units, and examples where applicable.
# 5. Be concise, practical, and actionable.
# 6. For treatments or chemical use, always mention safety steps and recommend consulting labels or local agricultural experts.
# 7. Stay on topic—focus on agriculture. Politely decline unrelated questions.
# 8. Reply in a clear manner, using paragraphs or lists where it improves readability.
# 9. ALWAYS RESPOND IN MARATHI LANGUAGE ONLY. Do not use English at all in your responses.
# </guidelines>
# """

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", chat_system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}"),
#         ]
#     )

#     basic_chat_chain = chat_prompt | llm

#     conversational_chain = RunnableWithMessageHistory(
#         basic_chat_chain,
#         get_session_history,
#         input_messages_key="input",
#         history_messages_key="chat_history",
#     )

#     try:
#         save_message(session_id, "human", query)
#         logger.info(f"Saved human message for session {session_id}")
#     except Exception as e:
#         logger.error(f"Failed to save human message for session {session_id}: {e}")

#     try:
#         ai_message = conversational_chain.invoke(
#             {"input": query}, config={"configurable": {"session_id": session_id}}
#         )
#         result = (
#             ai_message.content if hasattr(ai_message, "content") else str(ai_message)
#         )

#         logger.info(f"Generated AI response for session {session_id}")

#     except Exception as e:
#         logger.error(
#             f"Error invoking conversational chain for session {session_id}: {e}"
#         )
#         return jsonify({"error": "प्रतिसाद निर्माण करताना त्रुटी आली"}), 500  # Error message in Marathi

#     try:
#         save_message(session_id, "ai", result)
#         logger.info(f"Saved AI message for session {session_id}")
#     except Exception as e:
#         logger.error(f"Failed to save AI message for session {session_id}: {e}")

#     return jsonify({"response": result.replace("*", "")})


# engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
# Base = declarative_base()
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# class Prediction(Base):
#     __tablename__ = "predictions"
#     id = Column(Integer, primary_key=True, index=True)
#     prediction_id = Column(String, unique=True, index=True, nullable=False)
#     disease_name = Column(String, nullable=False)
#     confidence = Column(Float)
#     image_url = Column(String)
#     timestamp = Column(DateTime, default=datetime.now)


# class ChatHistory(Base):
#     __tablename__ = "chat_history"
#     id = Column(Integer, primary_key=True, index=True)
#     user_message = Column(Text, nullable=False)
#     bot_response = Column(Text, nullable=False)
#     disease_context = Column(String, nullable=True)
#     timestamp = Column(DateTime, default=datetime.now)


# @contextmanager
# def get_db_session():
#     """Provide a transactional scope around a series of operations."""
#     db = SessionLocal()
#     try:
#         yield db
#         db.commit()
#     except SQLAlchemyError as e:
#         db.rollback()
#         logger.error(f"Database Error: {e}")
#         raise
#     finally:
#         db.close()


# cloudinary.config(
#     cloud_name="dahksocqw",
#     api_key="774146768391685",
#     api_secret="j0T2SeocxZy_vTQ2KX-ClIygY0s",
# )

# GEMINI_API_KEY = "AIzaSyBCmTwvxa2-Vt7pMlRVgqX19bHj5628t3w"
# genai.configure(api_key=GEMINI_API_KEY)

# gemini_config = {
#     "temperature": 0.7,
#     "max_output_tokens": 500,
#     "top_p": 0.95,
#     "top_k": 40,
# }

# MODEL_FILENAME = "vgg16_multi_class_model_fine_tuned.h5"
# MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)
# GOOGLE_DRIVE_MODEL_ID =  "1xSdnUjcBFfY0emWcCKfxkoEqOW0i2VvR"


# def download_model_from_drive():
#     try:
#         logger.info("Downloading model from Google Drive...")

#         gdown.download(id=GOOGLE_DRIVE_MODEL_ID, output=MODEL_PATH, quiet=False)
#         logger.info(f"Model downloaded successfully to {MODEL_PATH}")
#         return True
#     except Exception as e:
#         logger.error(f"Error downloading model from Google Drive: {e}")
#         return False


# model = None


# def load_model():
#     global model
#     try:
#         if not os.path.exists(MODEL_PATH):
#             if not download_model_from_drive():
#                 return False
#         model = tf.keras.models.load_model(MODEL_PATH)
#         logger.info("Model loaded successfully.")
#         logger.info(f"Model input shape: {model.input_shape}")
#         return True
#     except Exception as e:
#         logger.error(f"Error loading model: {e}")
#         return False


# class_labels = [
#     "Apple__Apple_scab",
#     "Apple_Black_rot",
#     "Apple_Cedar_apple_rust",
#     "Apple_healthy",
#     "Blueberry_healthy",
#     "Brownspot",
#     "Cashew anthracnose",
#     "Cashew gumosis",
#     "Cashew healthy",
#     "Cashew leaf miner",
#     "Cashew red rust",
#     "Cassava bacterial blight",
#     "Cassava brown spot",
#     "Cassava green mite",
#     "Cassava healthy",
#     "Cassava mosaic",
#     "Cherry(including_sour)healthy",
#     "Cherry(including_sour)Powdery_mildew",
#     "Corn(maize)Cercospora_leaf_spot Gray_leaf_spot",
#     "Corn(maize)Common_rust",
#     "Corn_(maize)healthy",
#     "Corn(maize)Northern_Leaf_Blight",
#     "Cotton American Bollworm",
#     "Cotton Anthracnose",
#     "Cotton Aphid",
#     "Cotton Bacterial Blight",
#     "Cotton bollrot",
#     "Cotton bollworm",
#     "Cotton Healthy",
#     "cotton mealy bug",
#     "Cotton pink bollworm",
#     "Cotton thirps",
#     "cotton whitefly",
#     "Grape_Black_rot",
#     "Grape_Esca(Black_Measles)",
#     "Grape__healthy",
#     "Grape_Leaf_blight(Isariopsis_Leaf_Spot)",
#     "Gray_Leaf_Spot",
#     "Healthy Maize",
#     "Leaf Curl",
#     "Leaf smut",
#     "maize ear rot",
#     "Maize fall armyworm",
#     "Maize grasshoper",
#     "Maize healthy",
#     "Maize leaf beetle",
#     "Maize leaf blight",
#     "Maize leaf spot",
#     "maize stem borer",
#     "Maize streak virus",
#     "Orange__Haunglongbing(Citrus_greening)",
#     "Peach__Bacterial_spot",
#     "Peach_healthy",
#     "Pepper,_bell_Bacterial_spot",
#     "Pepper,_bell_healthy",
#     "Potato_Early_blight",
#     "Potato_healthy",
#     "Potato_Late_blight",
#     "Raspberry_healthy",
#     "red cotton bug",
#     "Rice Becterial Blight",
#     "Rice Blast",
#     "Rice Tungro",
#     "Soybean_healthy",
#     "Squash_Powdery_mildew",
#     "Strawberry_healthy",
#     "Strawberry_Leaf_scorch",
#     "Sugarcane Healthy",
#     "Sugarcane mosaic",
#     "Sugarcane RedRot",
#     "Sugarcane RedRust",
#     "Tomato leaf blight",
#     "Tomato leaf curl",
#     "Tomato verticulium wilt",
#     "Tomato_Bacterial_spot",
#     "Tomato_Early_blight",
#     "Tomato_healthy",
#     "Tomato_Late_blight",
#     "Tomato_Leaf_Mold",
#     "Tomato_Septoria_leaf_spot",
#     "Tomato_Spider_mites Two-spotted_spider_mite",
#     "Tomato_Target_Spot",
#     "Tomato_Tomato_mosaic_virus",
#     "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
#     "Wheat aphid",
#     "Wheat black rust",
#     "Wheat Brown leaf rust",
#     "Wheat Flag Smut",
#     "Wheat Healthy",
#     "Wheat leaf blight",
#     "Wheat mite",
#     "Wheat powdery mildew",
#     "Wheat scab",
#     "Wheat Stem fly",
#     "Wheat__Yellow_Rust",
#     "Wilt",
#     "Yellow Rust Sugarcane",
# ]

# HISTORY_FOLDER = "history"
# os.makedirs(HISTORY_FOLDER, exist_ok=True)


# @app.route("/")
# def home():
#     return render_template("cdr7.html")


# @app.route("/predict", methods=["POST"])
# def predict():
#     global model
#     if model is None:
#         if not load_model():
#             return (
#                 jsonify(
#                     {"error": "मॉडेल लोड करण्यात अयशस्वी. कृपया नंतर पुन्हा प्रयत्न करा."}  # Error in Marathi
#                 ),
#                 500,
#             )

#     if "file" not in request.files:
#         return jsonify({"error": "फाइल अपलोड केली नाही"}), 400  # Error in Marathi

#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "कोणतीही फाइल निवडली नाही"}), 400  # Error in Marathi

#     prediction_id = str(uuid.uuid4())
#     temp_filepath = os.path.join(HISTORY_FOLDER, f"{prediction_id}.jpg")

#     try:
#         file.save(temp_filepath)

#         image = tf.keras.preprocessing.image.load_img(
#             temp_filepath, target_size=(128, 128), color_mode="rgb"
#         )
#         input_arr = tf.keras.preprocessing.image.img_to_array(image)
#         input_arr = np.array([input_arr])
#         prediction_probs = model.predict(input_arr)
#         result_index = np.argmax(prediction_probs)
#         result = class_labels[result_index]
#         confidence = float(np.max(prediction_probs[0]))
#         logger.info(f"Prediction made: {result} with confidence {confidence:.4f}")

#         try:
#             cloudinary_response = cloudinary.uploader.upload(
#                 temp_filepath, folder="crop_disease", public_id=prediction_id
#             )
#             image_url = cloudinary_response["secure_url"]
#             logger.info(f"Image uploaded to Cloudinary: {image_url}")
#         except Exception as e:
#             logger.error(f"Error uploading to Cloudinary: {e}")
#             image_url = None

#         try:
#             with get_db_session() as db:
#                 new_prediction = Prediction(
#                     prediction_id=prediction_id,
#                     disease_name=result,
#                     confidence=confidence,
#                     image_url=image_url,
#                     timestamp=datetime.now(),
#                 )
#                 db.add(new_prediction)

#             logger.info(f"Prediction stored in SQLite with ID: {prediction_id}")
#         except Exception as e:
#             logger.error(f"Error storing prediction in SQLite: {e}")
#             return jsonify({"error": f"डेटा साठवण्यात त्रुटी: {e}"}), 500  # Error in Marathi

#         return jsonify({"prediction": result, "image_url": image_url})

#     except Exception as e:
#         logger.error(f"Error during prediction process: {e}")
#         return jsonify({"error": f"अंदाज प्रक्रियेत त्रुटी आली: {e}"}), 500  # Error in Marathi
#     finally:
#         if os.path.exists(temp_filepath):
#             try:
#                 os.remove(temp_filepath)
#                 logger.info(f"Removed temporary file: {temp_filepath}")
#             except OSError as rm_err:
#                 logger.error(f"Error removing temporary file {temp_filepath}: {rm_err}")



# # Add this new route to your Flask application
# # @app.route("/speech_to_text", methods=["POST"])
# # def speech_to_text():
# #     logger.info("Speech to text endpoint called")
    
# #     if 'audio' not in request.files:
# #         logger.error("No audio file provided")
# #         return jsonify({"error": "No audio file provided"}), 400

# #     audio_file = request.files['audio']
# #     session_id = request.form.get('session_id', 'default')
    
# #     if audio_file.filename == '':
# #         logger.error("No selected file")
# #         return jsonify({"error": "No selected file"}), 400

# #     try:
# #         # Create a temporary file to save the audio
# #         with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
# #             temp_audio_path = temp_audio.name
# #             audio_file.save(temp_audio_path)
        
# #         logger.info(f"Audio saved temporarily at {temp_audio_path}")
        
# #         # Convert to WAV using ffmpeg
# #         wav_path = temp_audio_path + ".wav"
# #         try:
# #             import subprocess
# #             subprocess.run(["ffmpeg", "-i", temp_audio_path, wav_path], check=True)
# #             logger.info(f"Audio converted to WAV: {wav_path}")
# #         except Exception as e:
# #             logger.error(f"Error converting audio to WAV: {e}")
# #             return jsonify({"error": "Failed to convert audio format"}), 500
        
# #         # Initialize recognizer
# #         recognizer = sr.Recognizer()
        
# #         # Convert the audio file to text
# #         with sr.AudioFile(wav_path) as source:
# #             logger.info("Reading audio file with recognizer")
# #             audio_data = recognizer.record(source)
            
# #             # Try recognition
# #             try:
# #                 # First try Hindi
# #                 try:
# #                     text = recognizer.recognize_google(audio_data, language='hi-IN')
# #                     logger.info(f"Audio recognized as Hindi: {text}")
# #                 except:
# #                     # Then try English
# #                     text = recognizer.recognize_google(audio_data, language='en-IN')
# #                     logger.info(f"Audio recognized as English: {text}")
# #             except sr.UnknownValueError:
# #                 logger.warning("Speech Recognition could not understand audio")
# #                 return jsonify({"error": "Could not understand audio"}), 400
# #             except sr.RequestError as e:
# #                 logger.error(f"Error with speech recognition service: {e}")
# #                 return jsonify({"error": "Speech recognition service error"}), 500
        
# #         # Clean up temporary files
# #         try:
# #             os.unlink(temp_audio_path)
# #             os.unlink(wav_path)
# #             logger.info("Temporary audio files removed")
# #         except Exception as e:
# #             logger.warning(f"Error removing temporary audio files: {e}")
        
# #         # Return the recognized text
# #         logger.info(f"Returning recognized text: {text}")
# #         return jsonify({"text": text})
    
# #     except Exception as e:
# #         logger.error(f"Error in speech to text conversion: {e}")
# #         return jsonify({"error": f"Error processing audio: {str(e)}"}), 500
# @app.route("/chatbot", methods=["POST"])
# def chatbot():
#     try:
#         data = request.json
#         user_message = data.get("message", "")
#         disease = data.get("disease", None)
        
#         # Updated prompts to include Marathi response instruction
#         if disease:
#             if user_message.startswith("Tell me about"):
#                 prompt = f"""You are an agricultural expert specializing in crop diseases. 
#                 Provide detailed information about {disease}, including its causes, symptoms, and effective remedies or treatments. 
#                 Format your response in a concise, easy-to-understand manner.
#                 IMPORTANT: YOUR RESPONSE MUST BE IN MARATHI LANGUAGE ONLY. DO NOT USE ANY ENGLISH."""
#             else:
#                 prompt = f"""You are an agricultural expert specializing in crop diseases. 
#                 The user previously asked about {disease} and now asks: '{user_message}'. 
#                 Provide a helpful, accurate response.
#                 IMPORTANT: YOUR RESPONSE MUST BE IN MARATHI LANGUAGE ONLY. DO NOT USE ANY ENGLISH."""
#         else:
#             prompt = f"""You are an agricultural expert specializing in crop diseases. 
#             A farmer asks: '{user_message}'. 
#             Provide a helpful, accurate response.
#             IMPORTANT: YOUR RESPONSE MUST BE IN MARATHI LANGUAGE ONLY. DO NOT USE ANY ENGLISH."""

#         try:
#             gemini_model = genai.GenerativeModel(
#                 "gemini-pro", generation_config=gemini_config
#             )
#             response = gemini_model.generate_content(prompt)
#             bot_response = response.text
#         except Exception as e:
#             logger.error(f"Error calling Gemini API: {e}")
#             bot_response = "मला सध्या माझ्या ज्ञानभांडारात प्रवेश करण्यात अडचण येत आहे. कृपया नंतर पुन्हा प्रयत्न करा."  # Error in Marathi

#         try:
#             with get_db_session() as db:
#                 chat_entry = ChatHistory(
#                     user_message=user_message,
#                     bot_response=bot_response,
#                     disease_context=disease,
#                     timestamp=datetime.now(),
#                 )
#                 db.add(chat_entry)
#             logger.info("Chat entry stored in SQLite.")
#         except Exception as e:
#             logger.error(f"Error storing chat history in SQLite: {e}")

#         return jsonify({"response": bot_response})

#     except Exception as e:
#         logger.error(f"Error in chatbot endpoint: {e}")
#         return (
#             jsonify(
#                 {
#                     "response": "क्षमस्व, मला एक त्रुटी आली. कृपया नंतर पुन्हा प्रयत्न करा."  # Error in Marathi
#                 }
#             ),
#             500,
#         )

# @app.route("/history")
# def history():
#     try:
#         with get_db_session() as db:
#             history_records = (
#                 db.query(Prediction).order_by(Prediction.timestamp.desc()).all()
#             )

#             history_data = [
#                 {
#                     "prediction_id": record.prediction_id,
#                     "disease_name": record.disease_name,
#                     "image_url": record.image_url,
#                     "timestamp": (
#                         record.timestamp.isoformat() if record.timestamp else None
#                     ),
#                 }
#                 for record in history_records
#             ]
#         return jsonify({"history": history_data})
#     except Exception as e:
#         logger.error(f"Error fetching history from SQLite: {e}")
#         return jsonify({"error": f"इतिहास प्राप्त करण्यात त्रुटी: {e}"}), 500  # Error in Marathi
    
# @app.route('/final', methods=['POST'])
# def handle_query():
#     data = request.get_json()
#     query = data.get('query', '')
#     session_id = data.get('session_id', '')
#     language = data.get('language', 'english')

#     print(f"Received Query: {query}")
#     print(f"Language: {language}")
    
#     # Add your logic here: you could handle Marathi differently if needed
#     if language == 'marathi':
#         response = f"तुमचं प्रश्न: {query}"  # simple echo
#     else:
#         response = f"Your query was: {query}"  # simple echo for English

#     return jsonify({'response': response})


# def create_db_tables():
#     try:
#         logger.info("Creating database tables if they don't exist...")
#         Base.metadata.create_all(bind=engine)
#         logger.info("Database tables checked/created.")
#     except Exception as e:
#         logger.error(f"Error creating database tables: {e}")


# if __name__ == "__main__":
#     create_db_tables()

#     if load_model():
#         logger.info("Starting Flask development server...")
#         app.run(debug=True, host="0.0.0.0", port=5000)
#     else:
#         logger.error("Failed to load model on startup. Application cannot start.")


