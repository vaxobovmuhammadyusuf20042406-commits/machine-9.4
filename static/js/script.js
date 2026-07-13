const predictionForm = document.getElementById("predictionForm");
const imageInput = document.getElementById("imageInput");
const uploadArea = document.getElementById("uploadArea");
const uploadPlaceholder = document.getElementById("uploadPlaceholder");
const imagePreview = document.getElementById("imagePreview");
const predictButton = document.getElementById("predictButton");
const formMessage = document.getElementById("formMessage");

const emptyResult = document.getElementById("emptyResult");
const resultContent = document.getElementById("resultContent");
const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceLevel = document.getElementById("confidenceLevel");
const modelName = document.getElementById("modelName");
const inferenceTime = document.getElementById("inferenceTime");
const speakButton = document.getElementById("speakButton");

const topFiveEmpty = document.getElementById("topFiveEmpty");
const topFiveList = document.getElementById("topFiveList");

const gradcamEmpty = document.getElementById("gradcamEmpty");
const gradcamContent = document.getElementById("gradcamContent");
const gradcamImage = document.getElementById("gradcamImage");

let latestPrediction = null;


function showMessage(text, type = "error") {
    formMessage.textContent = text;
    formMessage.className = `message ${type}`;
}


function hideMessage() {
    formMessage.textContent = "";
    formMessage.className = "message hidden";
}


function previewSelectedImage(file) {
    if (!file) {
        return;
    }

    if (!file.type.startsWith("image/")) {
        showMessage("Please choose a valid image file.");
        return;
    }

    const reader = new FileReader();

    reader.onload = function (event) {
        imagePreview.src = event.target.result;
        imagePreview.classList.remove("hidden");
        uploadPlaceholder.classList.add("hidden");
        hideMessage();
    };

    reader.readAsDataURL(file);
}


imageInput.addEventListener("change", function () {
    previewSelectedImage(imageInput.files[0]);
});


uploadArea.addEventListener("dragover", function (event) {
    event.preventDefault();
    uploadArea.classList.add("drag-active");
});


uploadArea.addEventListener("dragleave", function () {
    uploadArea.classList.remove("drag-active");
});


uploadArea.addEventListener("drop", function (event) {
    event.preventDefault();
    uploadArea.classList.remove("drag-active");

    const droppedFile = event.dataTransfer.files[0];

    if (!droppedFile) {
        return;
    }

    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(droppedFile);
    imageInput.files = dataTransfer.files;

    previewSelectedImage(droppedFile);
});


function updateTopFive(predictions) {
    topFiveList.innerHTML = "";

    predictions.forEach((item, index) => {
        const row = document.createElement("div");
        row.className = "top-five-row";

        row.innerHTML = `
            <div class="rank-number">${index + 1}</div>
            <div class="top-five-label">${item.label}</div>
            <div class="top-five-confidence">
                ${item.confidence.toFixed(2)}%
            </div>
        `;

        topFiveList.appendChild(row);
    });

    topFiveEmpty.classList.add("hidden");
    topFiveList.classList.remove("hidden");
}


function showPredictionResult(data) {
    latestPrediction = data;
    if (data.gradcam_url) {
    gradcamImage.src =
        `${data.gradcam_url}?timestamp=${Date.now()}`;

    gradcamEmpty.classList.add("hidden");
    gradcamContent.classList.remove("hidden");
}

    predictionText.textContent = data.prediction;
    confidenceText.textContent = `${data.confidence.toFixed(2)}%`;
    confidenceLevel.textContent = data.confidence_level;
    modelName.textContent = data.model;
    inferenceTime.textContent = `${data.inference_time_ms.toFixed(2)} ms`;

    confidenceBar.style.width = `${Math.min(data.confidence, 100)}%`;

    emptyResult.classList.add("hidden");
    resultContent.classList.remove("hidden");

    updateTopFive(data.top_5_predictions || []);
}


predictionForm.addEventListener("submit", async function (event) {
    event.preventDefault();
    hideMessage();

    const selectedFile = imageInput.files[0];

    if (!selectedFile) {
        showMessage("Please select an image before prediction.");
        return;
    }

    const formData = new FormData(predictionForm);

    predictButton.disabled = true;
    predictButton.textContent = "Analysing image...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(
                data.message || "Prediction could not be completed."
            );
        }

        showPredictionResult(data);
        showMessage(data.message, "success");
    } catch (error) {
        showMessage(error.message);
    } finally {
        predictButton.disabled = false;
        predictButton.textContent = "Predict object";
    }
});


speakButton.addEventListener("click", function () {
    if (!latestPrediction) {
        return;
    }

    if (!("speechSynthesis" in window)) {
        showMessage(
            "Text-to-speech is not supported in this browser."
        );
        return;
    }

    window.speechSynthesis.cancel();

    const message = new SpeechSynthesisUtterance(
        `The predicted object is ${latestPrediction.prediction}, ` +
        `with ${latestPrediction.confidence.toFixed(1)} percent confidence.`
    );

    message.rate = 0.95;
    message.pitch = 1;

    window.speechSynthesis.speak(message);
});