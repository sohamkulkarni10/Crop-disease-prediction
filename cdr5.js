// Variables to track chat state
let currentDisease = null;
let chatContainer = null;

function showContent(section) {
  const titleElement = document.getElementById("title");
  const contentElement = document.getElementById("content");
  const frontCropImage = document.getElementById("front-crop");
  const chatbotContainer = document.querySelector(".chatbot-container"); // Get the chatbot container

  // Reset visibility and content
  titleElement.style.display = "none";
  frontCropImage.style.display = "none";
  contentElement.innerHTML = "";
  contentElement.classList.remove("active");
  if (chatbotContainer) {
    chatbotContainer.classList.remove("active"); // Hide chatbot by default
  }

  if (section === "home") {
    contentElement.innerHTML = `
        <h2 class="welcome">Welcome to Home Page</h2>
        <img src="/static/crop.jpeg" alt="Crop Image" class="home-image">
        <p>Crop diseases are significant challenges in agriculture, impacting crop yield, quality, and overall food security worldwide. These diseases are caused by various pathogens, including fungi, bacteria, viruses, and nematodes, as well as environmental factors like nutrient deficiencies and unfavorable weather conditions. Common crop diseases include leaf spot, blight, rust, and wilt, which affect major crops like wheat, rice, maize, and vegetables. Early detection and accurate diagnosis are crucial for effective management and control, often involving the use of pesticides, crop rotation, resistant crop varieties, and advanced techniques like deep learning-based image analysis for disease prediction and prevention.

Crop diseases pose a severe threat to global agriculture, leading to substantial economic losses and jeopardizing food security. These diseases can affect plants at any stage of growth, reducing crop yield, quality, and market value. They are caused by various factors, including pathogens like fungi, bacteria, viruses, and nematodes, as well as abiotic stresses such as nutrient deficiencies, pollution, and climate extremes. For example, fungal diseases like powdery mildew and rust can spread rapidly under humid conditions, while bacterial diseases like blight and canker thrive in wet environments. Viral infections, such as the mosaic virus, are often transmitted by insect vectors, compounding the difficulty of controlling outbreaks.

Crop diseases can manifest as symptoms such as leaf discoloration, wilting, stunted growth, fruit rot, and even plant death. Staple crops like rice, wheat, maize, and potatoes are particularly vulnerable, with diseases like rice blast, wheat rust, and potato late blight causing devastating losses. Effective management requires an integrated approach, including crop rotation, use of resistant varieties, proper irrigation, and timely application of fungicides or insecticides. Modern advancements, such as machine learning and image-based diagnostic tools, enable farmers to detect diseases early, improving precision in treatment and minimizing environmental impact. By combining traditional agricultural practices with cutting-edge technology, the agricultural sector can better combat crop diseases and ensure sustainable food production.
                </p>
        `;
    contentElement.classList.add("active");
    // Show front page elements when Home is clicked
    titleElement.style.display = "block";
    frontCropImage.style.display = "block";
    document.body.style.overflowY = "auto"; // Enable scroll
  } else if (section === "crop") {
    contentElement.innerHTML = `
        <h2 class="crop_disease_prediction">Crop Disease Prediction</h2>
        <div class="upload-predict-container">
            <div class="upload-container">
                <label for="image-upload" class="upload-label">Upload Image:</label>
                <input type="file" id="image-upload" class="upload-input" accept="image/*">
                <div id="uploaded-image-preview" class="image-preview"></div>
            </div>
            <button class="predict-button">Predict</button>

             <div class="chatbot-container">
                  <label for="chatbot-input" class="chatbot-label">Ask something:</label>
                  <input type="text" id="chatbot-input" class="chatbot-input" placeholder="Type your message...">
                  <button id="send-button" class="chatbot-send-button">Send</button>
                  <div class="chat-messages"></div>
             </div>
        </div>
    `;
    contentElement.classList.add("active");
    if (chatbotContainer) {
      chatbotContainer.classList.add("active"); // Show chatbot on Crop Disease Recognition
      chatContainer = document.querySelector(".chat-messages"); // Initialize chatContainer

      const sendButton = document.getElementById("send-button");
      const chatbotInput = document.getElementById("chatbot-input");

      if (sendButton && chatbotInput && chatContainer) {
        sendButton.addEventListener("click", () => {
          const message = chatbotInput.value.trim();
          if (message) {
            displayMessage("You: " + message, "user-message");
            chatbotInput.value = "";
            getChatbotResponse(message, currentDisease).then((response) => {
              displayMessage("Bot: " + response, "bot-message");
            });
          }
        });

        chatbotInput.addEventListener("keypress", (event) => {
          if (event.key === "Enter") {
            sendButton.click();
          }
        });
      }
    }

    const imageUploadInput = document.getElementById("image-upload");
    const imagePreview = document.getElementById("uploaded-image-preview");
    const predictButton = document.querySelector(".predict-button");

    if (imageUploadInput) {
      imageUploadInput.addEventListener("change", function () {
        const file = imageUploadInput.files[0];
        if (file) {
          const reader = new FileReader();
          reader.onload = function (e) {
            imagePreview.innerHTML = `<img src="${e.target.result}" alt="Uploaded Image">`;
          };
          reader.readAsDataURL(file);
        } else {
          imagePreview.innerHTML = "";
        }
      });
    }

    if (predictButton) {
      predictButton.addEventListener("click", function () {
        const file = document.getElementById("image-upload").files[0];
        if (file) {
          const formData = new FormData();
          formData.append("file", file);

          fetch("/predict", {
            method: "POST",
            body: formData,
          })
            .then((response) => response.json())
            .then((data) => {
              console.log("prediction data:", data);
              contentElement.innerHTML += `<h2 class="predict-result">Prediction Result:</h2><p>${data.prediction}</p>`;

              // Update the chatbot with the predicted disease (commented out as per your request)
              currentDisease = data.prediction;

              // // Automatically send the disease to the chatbot (commented out as per your request)
              // displayMessage(
              //   "System: Disease detected - " + currentDisease,
              //   "system-message"
              // );

              // // Automatically get information about the disease (commented out as per your request)
              // getChatbotResponse(
              //   `Tell me about ${currentDisease}, its causes and remedies.`
              // ).then((response) => {
              //   displayMessage("Bot: " + response, "bot-message");
              // });
            })
            .catch((error) => {
              console.error("Error:", error);
              contentElement.innerHTML += `<p>An error occurred during prediction.</p>`;
            });
        } else {
          alert("Please upload an image first.");
        }
      });
    }
    document.body.style.overflowY = "auto"; // Enable scroll
  } else if (section === "history") {
    fetch("/history")
      .then((response) => response.json())
      .then((data) => {
        console.log("history data:", data);
        contentElement.innerHTML = `<h2 class="histroy-section">History of Crop Disease</h2>`;
        // Display the images and their predicted disease names horizontally
        const historyContainer = document.createElement("div");
        historyContainer.classList.add("history-item");
        data.history.forEach((item) => {
          historyContainer.innerHTML += `
                <div class="history-item-wrapper">
                    <img src="${item.image_url}" alt="History Image" width="150px">
                    <p class="disease-name">${item.disease_name}</p>
                </div>
            `;
        });
        contentElement.appendChild(historyContainer);
        contentElement.classList.add("active");
      })
      .catch((error) => {
        console.error("Error fetching history:", error);
        contentElement.innerHTML += `<p>Unable to fetch history images.</p>`;
        contentElement.classList.add("active");
      });
    document.body.style.overflowY = "auto";
  } else if (section === "Dashboard") {
    contentElement.innerHTML = `<h2 class="dashboard-section">Dashboard</h2><div id="reportContainer"></div>`;
    contentElement.classList.add("active");
    fetch("/getEmbedInfo")
      .then((response) => response.json())
      .then((embedInfo) => {
        // Load Power BI report
        powerbi.load({
          id: embedInfo.reportId,
          embedUrl: embedInfo.embedUrl,
          accessToken: embedInfo.accessToken,
          type: "report",
          tokenType: powerbi.models.TokenType.Embed,
          settings: {
            navContentPaneEnabled: false,
            filterPaneEnabled: false,
          },
        });
      })
      .catch((error) => {
        console.error("Error fetching embed info:", error);
        document.getElementById("reportContainer").innerText =
          "Failed to load dashboard.";
      });
    document.body.style.overflowY = "auto";
  } else {
    document.body.style.overflowY = "hidden";
  }
}

function displayMessage(message, className) {
  if (chatContainer) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", className);
    messageDiv.textContent = message;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight; // Scroll to the latest message
  }
}

async function getChatbotResponse(message, disease = null) {
  try {
    const response = await fetch("/chatbot", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: message, disease: disease }),
    });
    const data = await response.json();
    return data.response;
  } catch (error) {
    console.error("Error fetching chatbot response:", error);
    return "Sorry, I couldn't get a response.";
  }
}

// Set initial scroll behavior
// document.addEventListener("DOMContentLoaded", () => {
//   showContent(null); // Set initial state (no scroll)
// });
// Show home page on initial load
// document.addEventListener("DOMContentLoaded", () => {
//   showContent("home");
// });
