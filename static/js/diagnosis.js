// ✅ Toggle Menu Function
function toggleMenu() {
    var menu = document.getElementById("dropdownMenu");
    menu.style.display = menu.style.display === "block" ? "none" : "block";
}

document.addEventListener("DOMContentLoaded", function () {
    let currentIndex = 0;
    const slides = document.querySelectorAll(".slide");
    const totalSlides = slides.length;
    const slider = document.querySelector(".slider");
    const indicators = document.querySelectorAll(".indicator");

    function updateSlidePosition() {
        slider.style.transform = `translateX(-${currentIndex * 100}%)`;

        // Update Active Indicator
        indicators.forEach((dot, index) => {
            dot.classList.toggle("active", index === currentIndex);
        });
    }

    function nextSlide() {
        currentIndex = (currentIndex + 1) % totalSlides;
        updateSlidePosition();
    }

    function prevSlide() {
        currentIndex = (currentIndex - 1 + totalSlides) % totalSlides;
        updateSlidePosition();
    }

    // Go to Specific Slide when Clicking on Indicator
    function goToSlide(index) {
        currentIndex = index;
        updateSlidePosition();
    }

    // Attach event listeners to buttons
    document.querySelector(".prev").addEventListener("click", prevSlide);
    document.querySelector(".next").addEventListener("click", nextSlide);
    
    // Attach event listeners to indicators
    indicators.forEach((dot, index) => {
        dot.addEventListener("click", () => goToSlide(index));
    });

    // ✅ Ensure first slide & indicator are correct at start
    updateSlidePosition();

    // ✅ Auto-slide every 3 seconds (without changing manual controls)
    setInterval(nextSlide, 3000);
});

function clearFile(inputId) {
    document.getElementById(inputId).value = "";
}



// frontend and backend communication
// function startDiagnosis(inputId, fileNameId, type, loadingId, resultId, outputId) {
    // let fileInput = document.getElementById(inputId);
    // let fileNameDisplay = document.getElementById(fileNameId);
    // let loadingText = document.getElementById(loadingId);
    // let resultText = document.getElementById(resultId);
    // let outputImage = document.getElementById(outputId);
// 
    // Ensure a file is selected
    // if (!fileInput.files.length) {
        // alert("Please upload an image file first.");
        // return;
    // }
// 
    // let file = fileInput.files[0];
    // let formData = new FormData();
    // formData.append("file", file);
    // 
    // Determine image type (1 = X-ray, 2 = MRI)
    // formData.append("type", type === 1 ? "xray" : "mri");
// 
    // Update UI
    // fileNameDisplay.textContent = "File: " + file.name;
    // loadingText.style.display = "block";
    // resultText.style.display = "none";
    // outputImage.style.display = "none";
// 
    // Send file to Flask API
    // fetch("/predict", {
        // method: "POST",
        // body: formData
    // })
    // .then(response => response.json())
    // .then(data => {
        // loadingText.style.display = "none"; // Hide loading
        // if (data.error) {
            // resultText.textContent = "Error: " + data.error;
        // } else {
            // resultText.innerHTML = `Prediction: <strong>${data.prediction}</strong><br>Confidence: <strong>${(data.confidence * 100).toFixed(2)}%</strong>`;
            // outputImage.src = data.plot_path; // Show probability plot
            // outputImage.style.display = "block";
        // }
        // resultText.style.display = "block";
    // })
    // .catch(error => {
        // console.error("Error:", error);
        // resultText.textContent = "Error occurred. Please try again.";
        // resultText.style.display = "block";
        // loadingText.style.display = "none";
    // });
// }


// frontend and backend communication( for 4 diseases)
function startDiagnosis(inputId, fileNameId, type, loadingId, resultId, outputId) {
    let fileInput = document.getElementById(inputId);
    let fileNameDisplay = document.getElementById(fileNameId);
    let loadingText = document.getElementById(loadingId);
    let resultText = document.getElementById(resultId);
    let outputImage = document.getElementById(outputId);

    // Ensure a file is selected
    if (!fileInput.files.length) {
        alert("Please upload an image file first.");
        return;
    }

    let file = fileInput.files[0];
    let formData = new FormData();
    formData.append("file", file);
    
    // Determine model type (Corrected)
    let modelType = "";
    switch (type) {
        case 1:
            modelType = "xray";
            break;
        case 2:
            modelType = "mri";
            break;
        case 3:
            modelType = "retina";
            break;
        case 4:
            modelType = "kidney_stone";
            break;
        default:
            alert("Invalid type specified.");
            return;
    }
    formData.append("type", modelType);  // Send correct type to Flask

    // Update UI
    fileNameDisplay.textContent = "File: " + file.name;
    loadingText.style.display = "block";
    resultText.style.display = "none";
    outputImage.style.display = "none";

    // Send file to Flask API
    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loadingText.style.display = "none"; // Hide loading
        if (data.error) {
            resultText.textContent = "Error: " + data.error;
        } else {
            resultText.innerHTML = `Prediction: <strong>${data.prediction}</strong><br>Confidence: <strong>${(data.confidence * 100).toFixed(2)}%</strong>`;
            outputImage.src = data.plot_path; // Show probability plot
            outputImage.style.display = "block";
        }
        resultText.style.display = "block";
    })
    .catch(error => {
        console.error("Error:", error);
        resultText.textContent = "Error occurred. Please try again.";
        resultText.style.display = "block";
        loadingText.style.display = "none";
    });
}
