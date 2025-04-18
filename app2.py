# from flask import Flask, render_template, request, jsonify, send_from_directory
# import os
# import uuid
# import numpy as np
# from tensorflow import keras
# from PIL import Image
# import logging
# import cloudinary
# import cloudinary.uploader
# import pymongo
# from pymongo import MongoClient
# from datetime import datetime
# import requests
# from io import BytesIO
# import tensorflow as tf
# app = Flask(__name__)

# # Configure Cloudinary
# cloudinary.config(
#     cloud_name="dahksocqw",  # Replace with your Cloudinary credentials
#     api_key="774146768391685",
#     api_secret="j0T2SeocxZy_vTQ2KX-ClIygY0s"
# )

# # Configure MongoDB
# client = MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB connection string
# db = client["crop_disease_db"]
# predictions_collection = db["predictions"]

# # Load trained model
# MODEL_PATH = "vgg16_multi_class_model_fine_tuned.h5"
# model = None



# try:
#     model = keras.models.load_model(MODEL_PATH)
#     print("success")
#     print(model.input_shape)  # Check expected shape
#     logging.info("Model loaded successfully.")
# except Exception as e:
#     logging.error(f"Error loadingmodel: {e}")
    
# class_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Brownspot', 'Cashew anthracnose', 'Cashew gumosis', 'Cashew healthy', 'Cashew leaf miner', 'Cashew red rust', 'Cassava bacterial blight', 'Cassava brown spot', 'Cassava green mite', 'Cassava healthy', 'Cassava mosaic', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Cotton American Bollworm', 'Cotton Anthracnose', 'Cotton Aphid', 'Cotton Bacterial Blight', 'Cotton bollrot', 'Cotton bollworm', 'Cotton Healthy', 'cotton mealy bug', 'Cotton pink bollworm', 'Cotton thirps', 'cotton whitefly', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Gray_Leaf_Spot', 'Healthy Maize', 'Leaf Curl', 'Leaf smut', 'maize ear rot', 'Maize fall armyworm', 'Maize grasshoper', 'Maize healthy', 'Maize leaf beetle', 'Maize leaf blight', 'Maize leaf spot', 'maize stem borer', 'Maize streak virus', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'red cotton bug', 'Rice Becterial Blight', 'Rice Blast', 'Rice Tungro', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Sugarcane Healthy', 'Sugarcane mosaic', 'Sugarcane RedRot', 'Sugarcane RedRust', 'Tomato leaf blight', 'Tomato leaf curl', 'Tomato verticulium wilt', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Wheat aphid', 'Wheat black rust', 'Wheat Brown leaf rust', 'Wheat Flag Smut', 'Wheat Healthy', 'Wheat leaf blight', 'Wheat mite', 'Wheat powdery mildew', 'Wheat scab', 'Wheat Stem fly', 'Wheat___Yellow_Rust', 'Wilt', 'Yellow Rust Sugarcane']
    
    


# # Ensure history folder exists
# HISTORY_FOLDER = "history"
# os.makedirs(HISTORY_FOLDER, exist_ok=True)

# # Serve static files (css, js)
# @app.route('/')
# def home():
#     return render_template("cdr7.html")

# @app.route('/cdr7.css')
# def serve_css():
#     return send_from_directory('./static/', 'cdr5.css')

# @app.route('/cdr7.js')
# def serve_js():
#     return send_from_directory('./static/', 'cdr5.js')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     # Generate a unique ID for this prediction
#     prediction_id = str(uuid.uuid4())
    
#     # Save file temporarily
#     temp_filepath = os.path.join(HISTORY_FOLDER, f"{prediction_id}.jpg")
#     file.save(temp_filepath)

#     # # Load and preprocess image
#     # try:
#     #     img = Image.open(temp_filepath)
#     #     img = img.resize((128, 128))  # Resize based on model requirement
#     #     img_array = np.array(img) / 255.0   # Normalize
#     #     input_arr = np.expand_dims(img_array, axis=0)

#     #     logging.info(f"Image loaded and preprocessed for prediction: {temp_filepath}")
#     # except Exception as e:
#     #     logging.error(f"Error processing image: {e}")
#     #     return jsonify({"error": "Error processing image."}), 500

#     # # Predict class
#     # try:
#     #     prediction = model.predict(input_arr)
#     #     predicted_class = np.argmax(prediction)  # Extract the highest probability class index
#     #     print('---------------------------------------',predicted_class)
        
#     #     result = class_labels[predicted_class]
#     #     confidence = float(np.max(prediction))  
       
        
        
#     # except Exception as e:
#     #     logging.error(f"Error making prediction: {e}")
#     #     return jsonify({"error": f"Error making prediction: {e}"}), 500

#     model = tf.keras.models.load_model('vgg16_multi_class_model_fine_tuned.h5')
#     image = tf.keras.preprocessing.image.load_img(temp_filepath, target_size=(128, 128), color_mode="rgb")
#     input_arr = tf.keras.preprocessing.image.img_to_array(image)
#     input_arr = np.array([input_arr])
#     prediction = model.predict(input_arr)
#     result_index = np.argmax(prediction)
#     result = class_labels[result_index]
#     logging.info(f"Prediction made: {result}")
        
#     # Upload to Cloudinary
#     try:
#         cloudinary_response = cloudinary.uploader.upload(temp_filepath, 
#                                                        folder="crop_disease",
#                                                        public_id=prediction_id)
#         image_url = cloudinary_response['secure_url']
#         logging.info(f"Image uploaded to Cloudinary: {image_url}")
#     except Exception as e:
#         logging.error(f"Error uploading to Cloudinary: {e}")
#         return jsonify({"error": f"Error storing image: {e}"}), 500

#     # Store in MongoDB
#     try:
#         prediction_data = {
#             "prediction_id": prediction_id,
#             "disease_name": result,
#             "confidence": float(np.max(prediction[0])),  # Store highest confidence score
#             "image_url": image_url,
#             "timestamp": datetime.now()
#         }
#         predictions_collection.insert_one(prediction_data)
#         logging.info(f"Prediction stored in MongoDB with ID: {prediction_id}")
#     except Exception as e:
#         logging.error(f"Error storing in MongoDB: {e}")
#         return jsonify({"error": f"Error storing prediction data: {e}"}), 500
    
#     # We can remove the temporary file now
#     os.remove(temp_filepath)

#     # Return the prediction result and image URL
#     return jsonify({
#         "prediction": result,
#         "image_url": image_url
#     })

# @app.route('/history')
# def history():
#     try:
#         # Fetch history from MongoDB instead of local files
#         history_data = list(predictions_collection.find({}, {
#             "_id": 0,
#             "prediction_id": 1,
#             "disease_name": 1,
#             "image_url": 1,
#             "timestamp": 1
#         }).sort("timestamp", pymongo.DESCENDING))
        
#         # Convert datetime to string for JSON serialization
#         for item in history_data:
#             if "timestamp" in item:
#                 item["timestamp"] = item["timestamp"].isoformat()
#         print('data====================>',history_data)
#         return jsonify({"history": history_data})
#     except Exception as e:
#         logging.error(f"Error fetching history from MongoDB: {e}")
#         return jsonify({"error": f"Error fetching history: {e}"}), 500
    
    


# if __name__ == '__main__':
#     app.run(debug=True)     


from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
import numpy as np
from tensorflow import keras
from PIL import Image
import logging
import cloudinary
import cloudinary.uploader
import pymongo
from pymongo import MongoClient
from datetime import datetime
import requests
from io import BytesIO
import tensorflow as tf
import openai  # Import OpenAI library
import json

app = Flask(__name__)

# Configure OpenAI (Replace with your actual API key)
openai.api_key = "YOUR_OPENAI_API_KEY_HERE"

TENANT_ID = 'f057284d-5d7d-4b6f-b94c-707b3dd79c3d'
CLIENT_ID = '91d2bb66-a0c6-489c-9aa7-10ea8f3e410f'
CLIENT_SECRET = 'FIf8Q~h6QWj5wFNF8A_SZZfm-azq6sjxmU9lnctr'
WORKSPACE_ID = 'ad5b0bc7-3d0b-4042-911a-6cec44b4367b'
REPORT_ID = 'dd6bb935-f692-436f-98e4-e4467dd8598e'

AUTHORITY_URL = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
RESOURCE_URL = 'https://api.powerbi.com/'

@app.route('/getEmbedInfo', methods=['GET'])
def get_embed_info():
    try:
        # Step 1: Get Azure AD Token
        token_response = requests.post(AUTHORITY_URL, data={
            'grant_type': 'client_credentials',
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'scope': 'https://analysis.windows.net/powerbi/api/.default'
        })

        token_response.raise_for_status()
        aad_token = token_response.json()['access_token']

        # Step 2: Generate Embed Token for Report
        embed_token_url = f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}/reports/{REPORT_ID}/GenerateToken"
        embed_body = {
            "accessLevel": "View"
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {aad_token}'
        }

        embed_token_response = requests.post(embed_token_url, headers=headers, json=embed_body)
        embed_token_response.raise_for_status()

        embed_token = embed_token_response.json().get('token')
        embed_url = f"https://app.powerbi.com/reportEmbed?reportId={REPORT_ID}&groupId={WORKSPACE_ID}"

        return jsonify({
            'accessToken': embed_token,
            'embedUrl': embed_url,
            'reportId': REPORT_ID
        })

    except Exception as e:
        print(e)
        return jsonify({'error': 'Failed to get embed token'}), 500





# Configure OpenAI (Replace with your API key)
# openai.api_key = "sk-abcdef1234567890abcdef1234567890abcdef12"

# Configure Cloudinary
cloudinary.config(
    cloud_name="dahksocqw",
    api_key="774146768391685",
    api_secret="j0T2SeocxZy_vTQ2KX-ClIygY0s"
)

# Configure MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["crop_disease_db"]
predictions_collection = db["predictions"]
chat_history_collection = db["chat_history"]  # New collection for chat history

# Load trained model
MODEL_PATH = "vgg16_multi_class_model_fine_tuned.h5"
model = None

try:
    model = keras.models.load_model(MODEL_PATH)
    print("success")
    print(model.input_shape)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    
class_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Brownspot', 'Cashew anthracnose', 'Cashew gumosis', 'Cashew healthy', 'Cashew leaf miner', 'Cashew red rust', 'Cassava bacterial blight', 'Cassava brown spot', 'Cassava green mite', 'Cassava healthy', 'Cassava mosaic', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Cotton American Bollworm', 'Cotton Anthracnose', 'Cotton Aphid', 'Cotton Bacterial Blight', 'Cotton bollrot', 'Cotton bollworm', 'Cotton Healthy', 'cotton mealy bug', 'Cotton pink bollworm', 'Cotton thirps', 'cotton whitefly', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Gray_Leaf_Spot', 'Healthy Maize', 'Leaf Curl', 'Leaf smut', 'maize ear rot', 'Maize fall armyworm', 'Maize grasshoper', 'Maize healthy', 'Maize leaf beetle', 'Maize leaf blight', 'Maize leaf spot', 'maize stem borer', 'Maize streak virus', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'red cotton bug', 'Rice Becterial Blight', 'Rice Blast', 'Rice Tungro', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Sugarcane Healthy', 'Sugarcane mosaic', 'Sugarcane RedRot', 'Sugarcane RedRust', 'Tomato leaf blight', 'Tomato leaf curl', 'Tomato verticulium wilt', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Wheat aphid', 'Wheat black rust', 'Wheat Brown leaf rust', 'Wheat Flag Smut', 'Wheat Healthy', 'Wheat leaf blight', 'Wheat mite', 'Wheat powdery mildew', 'Wheat scab', 'Wheat Stem fly', 'Wheat___Yellow_Rust', 'Wilt', 'Yellow Rust Sugarcane']

# Ensure history folder exists
HISTORY_FOLDER = "history"
os.makedirs(HISTORY_FOLDER, exist_ok=True)

# Serve static files (css, js)
@app.route('/')
def home():
    return render_template("cdr7.html")

@app.route('/cdr7.css')
def serve_css():
    return send_from_directory('./static/', 'cdr5.css')

@app.route('/cdr7.js')
def serve_js():
    return send_from_directory('./static/', 'cdr5.js')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Generate a unique ID for this prediction
    prediction_id = str(uuid.uuid4())
    
    # Save file temporarily
    temp_filepath = os.path.join(HISTORY_FOLDER, f"{prediction_id}.jpg")
    file.save(temp_filepath)

    # Load and predict using the model
    try:
        model = tf.keras.models.load_model('vgg16_multi_class_model_fine_tuned.h5')
        image = tf.keras.preprocessing.image.load_img(temp_filepath, target_size=(128, 128), color_mode="rgb")
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        result = class_labels[result_index]
        logging.info(f"Prediction made: {result}")
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return jsonify({"error": f"Error making prediction: {e}"}), 500
        
    # Upload to Cloudinary
    try:
        cloudinary_response = cloudinary.uploader.upload(temp_filepath, 
                                                       folder="crop_disease",
                                                       public_id=prediction_id)
        image_url = cloudinary_response['secure_url']
        logging.info(f"Image uploaded to Cloudinary: {image_url}")
    except Exception as e:
        logging.error(f"Error uploading to Cloudinary: {e}")
        return jsonify({"error": f"Error storing image: {e}"}), 500

    # Store in MongoDB
    try:
        prediction_data = {
            "prediction_id": prediction_id,
            "disease_name": result,
            "confidence": float(np.max(prediction[0])),
            "image_url": image_url,
            "timestamp": datetime.now()
        }
        predictions_collection.insert_one(prediction_data)
        logging.info(f"Prediction stored in MongoDB with ID: {prediction_id}")
    except Exception as e:
        logging.error(f"Error storing in MongoDB: {e}")
        return jsonify({"error": f"Error storing prediction data: {e}"}), 500
    
    # Remove the temporary file
    os.remove(temp_filepath)

    # Return the prediction result and image URL
    return jsonify({
        "prediction": result,
        "image_url": image_url
    })

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.json
        user_message = data.get('message', '')
        disease = data.get('disease', None)
        
        # Create a context-aware prompt
        if disease:
            # If this is the first message after disease detection
            if user_message.startswith("Tell me about"):
                prompt = f"You are an agricultural expert specializing in crop diseases. Provide detailed information about {disease}, including its causes, symptoms, and effective remedies or treatments. Format your response in a concise, easy-to-understand manner."
            else:
                # For follow-up questions
                prompt = f"You are an agricultural expert specializing in crop diseases. The user previously asked about {disease} and now asks: '{user_message}'. Provide a helpful, accurate response."
        else:
            # General crop disease questions
            prompt = f"You are an agricultural expert specializing in crop diseases. A farmer asks: '{user_message}'. Provide a helpful, accurate response."
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can use "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are an agricultural expert specializing in crop diseases."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # Extract the response
        bot_response = response.choices[0].message.content.strip()
        
        # Store the conversation in MongoDB
        chat_entry = {
            "user_message": user_message,
            "bot_response": bot_response,
            "disease_context": disease,
            "timestamp": datetime.now()
        }
        chat_history_collection.insert_one(chat_entry)
        
        return jsonify({"response": bot_response})
    
    except Exception as e:
        logging.error(f"Error in chatbot: {e}")
        return jsonify({"response": "I'm sorry, I encountered an error. Please try again later."}), 500



# @app.route('/chatbot', methods=['POST'])
# def chatbot():
#     try:
#         data = request.json
#         user_message = data.get('message', '')
#         disease = data.get('disease', None)
        
#         # Create a context-aware prompt
#         if disease:
#             # If this is the first message after disease detection
#             if user_message.startswith("Tell me about"):
#                 prompt = f"You are an agricultural expert specializing in crop diseases. Provide detailed information about {disease}, including its causes, symptoms, and effective remedies or treatments. Format your response in a concise, easy-to-understand manner."
#             else:
#                 # For follow-up questions
#                 prompt = f"You are an agricultural expert specializing in crop diseases. The user previously asked about {disease} and now asks: '{user_message}'. Provide a helpful, accurate response."
#         else:
#             # General crop disease questions
#             prompt = f"You are an agricultural expert specializing in crop diseases. A farmer asks: '{user_message}'. Provide a helpful, accurate response."
        
#         # Call OpenAI API
#         response = openai.ChatCompletion.create(
#             model="gpt-4",  # You can change to gpt-3.5-turbo if needed
#             messages=[
#                 {"role": "system", "content": "You are an agricultural expert specializing in crop diseases."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=500,
#             temperature=0.7
#         )
        
#         # Extract the response
#         bot_response = response.choices[0].message.content.strip()
        
#         # Store the conversation in MongoDB
#         chat_entry = {
#             "user_message": user_message,
#             "bot_response": bot_response,
#             "disease_context": disease,
#             "timestamp": datetime.now()
#         }
#         chat_history_collection.insert_one(chat_entry)
        
#         return jsonify({"response": bot_response})
    
#     except Exception as e:
#         logging.error(f"Error in chatbot: {e}")
#         return jsonify({"response": "I'm sorry, I encountered an error. Please try again later."}), 500




@app.route('/history')
def history():
    try:
        # Fetch history from MongoDB
        history_data = list(predictions_collection.find({}, {
            "_id": 0,
            "prediction_id": 1,
            "disease_name": 1,
            "image_url": 1,
            "timestamp": 1
        }).sort("timestamp", pymongo.DESCENDING))
        
        # Convert datetime to string for JSON serialization
        for item in history_data:
            if "timestamp" in item:
                item["timestamp"] = item["timestamp"].isoformat()
        
        return jsonify({"history": history_data})
    except Exception as e:
        logging.error(f"Error fetching history from MongoDB: {e}")
        return jsonify({"error": f"Error fetching history: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
    
from flask import Flask, jsonify
import requests

app = Flask(__name__)


# # Replace with your actual details
# TENANT_ID = 'f057284d-5d7d-4b6f-b94c-707b3dd79c3d'
# CLIENT_ID = '91d2bb66-a0c6-489c-9aa7-10ea8f3e410f'
# CLIENT_SECRET = 'FIf8Q~h6QWj5wFNF8A_SZZfm-azq6sjxmU9lnctr'
# SCOPE = 'https://analysis.windows.net/powerbi/api/.default'

# AUTHORITY_URL = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"

# @app.route('/getAccessToken', methods=['GET'])
# def get_access_token():
#     headers = {
#         'Content-Type': 'application/x-www-form-urlencoded'
#     }
#     body = {
#         'grant_type': 'client_credentials',
#         'client_id': CLIENT_ID,
#         'client_secret': CLIENT_SECRET,
#         'scope': SCOPE
#     }

#     try:
#         response = requests.post(AUTHORITY_URL, headers=headers, data=body)
#         response.raise_for_status()

#         access_token = response.json().get('access_token')
#         return jsonify({'accessToken': access_token})

#     except requests.exceptions.RequestException as e:
#         print(e)
#         return jsonify({'error': 'Failed to get access token'}), 500

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000, debug=True)