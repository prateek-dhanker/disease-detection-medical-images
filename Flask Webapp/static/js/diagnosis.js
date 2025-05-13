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
// frontend and backend communication (for 4 diseases)
function startDiagnosis(inputId, fileNameId, type, loadingId, resultId, outputId, showResultId) {
    let fileInput = document.getElementById(inputId);
    let fileNameDisplay = document.getElementById(fileNameId);
    let loadingText = document.getElementById(loadingId);
    let resultText = document.getElementById(resultId);
    let outputImage = document.getElementById(outputId);
    let showResultBtn = document.getElementById(showResultId); // Ensure this targets the right button

    // Ensure a file is selected
    if (!fileInput.files.length) {
        alert("Please upload an image file first.");
        return;
    }

    let file = fileInput.files[0];
    let formData = new FormData();
    formData.append("file", file);
    
    // Determine model type
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
        case 5:
            modelType = "bone";
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
    showResultBtn.style.display = "none"; // Hide button initially

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
            resultText.style.display = "block";
        } else {
            // Store result in sessionStorage for the next page
            sessionStorage.setItem("prediction", data.prediction);
            sessionStorage.setItem("confidence", (data.confidence * 100).toFixed(2));
            sessionStorage.setItem("plot_path", data.plot_path);

            // Ensure the correct Show Result button appears
            showResultBtn.style.display = "block";
        }
    })
    .catch(error => {
        console.error("Error:", error);
        resultText.textContent = "Error occurred. Please try again.";
        resultText.style.display = "block";
        loadingText.style.display = "none";
    });

    // Button click event to open result page
    showResultBtn.onclick = function() {
        window.location.href = "/result";  // Redirect to Flask route
    };
}


// Disease Information Database
const diseaseData = {
    "COVID-19": {
        title: "COVID-19",
        causes: "Caused by the SARS-CoV-2 virus, spreads through respiratory droplets.",
        symptoms: ["Fever", "Cough", "Shortness of breath", "Loss of taste or smell"],
        stages: [
            "Stage 1: Mild fever and cough",
            "Stage 2: Breathing difficulties, pneumonia symptoms",
            "Stage 3: Severe lung inflammation, organ failure"
        ],
        emergencyMedicine: ["Paracetamol", "Oxygen therapy", "Antiviral drugs"]
    },
    "Pneumonia": {
        title: "Pneumonia",
        causes: "Infection caused by bacteria, viruses, or fungi affecting the lungs.",
        symptoms: ["Chest pain", "Fever", "Shortness of breath", "Fatigue"],
        stages: [
            "Stage 1: Mild cough and fever",
            "Stage 2: Difficulty breathing, chest congestion",
            "Stage 3: Severe infection, hospitalization needed"
        ],
        emergencyMedicine: ["Antibiotics (if bacterial)", "Oxygen therapy"]
    },
    "Tuberculosis": {
        title: "Tuberculosis (TB)",
        causes: "Caused by Mycobacterium tuberculosis bacteria, spreads through air.",
        symptoms: ["Chronic cough", "Weight loss", "Night sweats", "Fatigue"],
        stages: [
            "Stage 1: Latent TB (no symptoms)",
            "Stage 2: Active TB (persistent cough, fever)",
            "Stage 3: Advanced TB (lung damage, potential spreading to other organs)"
        ],
        emergencyMedicine: ["Rifampin", "Isoniazid", "Ethambutol"]
    },
    "Cataract": {
        title: "Cataract",
        causes: "Clouding of the eye lens due to aging, diabetes, or UV exposure.",
        symptoms: ["Blurred vision", "Sensitivity to light", "Difficulty seeing at night"],
        stages: [
            "Stage 1: Slight blurring of vision",
            "Stage 2: Increasing difficulty in night vision",
            "Stage 3: Complete clouding, potential blindness if untreated"
        ],
        emergencyMedicine: ["Surgery is the only treatment"]
    },
    "Diabetic Retinopathy": {
        title: "Diabetic Retinopathy",
        causes: "Caused by prolonged high blood sugar levels damaging eye blood vessels.",
        symptoms: ["Blurred vision", "Dark spots in vision", "Difficulty seeing colors"],
        stages: [
            "Stage 1: Mild swelling in retina",
            "Stage 2: Severe blood vessel damage",
            "Stage 3: Vision loss due to retina detachment"
        ],
        emergencyMedicine: ["Laser surgery", "Anti-VEGF injections"]
    },
    "Brain Tumor": {
        title: "Brain Tumor",
        causes: "Abnormal growth of brain cells, can be cancerous or non-cancerous.",
        symptoms: ["Headaches", "Seizures", "Vision problems", "Memory loss"],
        stages: [
            "Stage 1: Mild symptoms (headache, dizziness)",
            "Stage 2: Increased pressure in brain, cognitive impairment",
            "Stage 3: Severe neurological damage, potential death"
        ],
        emergencyMedicine: ["Surgery", "Radiation therapy", "Chemotherapy"]
    },
    "Spinal Cord Injury": {
        title: "Spinal Cord Injury",
        causes: "Damage to the spinal cord due to trauma or disease.",
        symptoms: ["Loss of sensation", "Paralysis", "Breathing difficulty"],
        stages: [
            "Stage 1: Loss of movement and pain",
            "Stage 2: Nerve damage, loss of bladder control",
            "Stage 3: Permanent paralysis in severe cases"
        ],
        emergencyMedicine: ["Steroids", "Surgery", "Physical therapy"]
    },
    "Kidney Stones": {
        title: "Kidney Stones",
        causes: "Hard mineral deposits in kidneys due to dehydration or diet.",
        symptoms: ["Severe pain in lower back", "Blood in urine", "Nausea"],
        stages: [
            "Stage 1: Small stones causing minor discomfort",
            "Stage 2: Stones increasing in size, blocking urine flow",
            "Stage 3: Severe pain, possible kidney damage"
        ],
        emergencyMedicine: ["Pain relievers", "Hydration", "Lithotripsy (shock wave therapy)"]
    },
    "Kidney Infection": {
        title: "Kidney Infection",
        causes: "Bacterial infection affecting the kidneys.",
        symptoms: ["Fever", "Back pain", "Frequent urination", "Fatigue"],
        stages: [
            "Stage 1: Mild infection causing discomfort",
            "Stage 2: Increased fever, pain, and nausea",
            "Stage 3: Sepsis risk if untreated"
        ],
        emergencyMedicine: ["Antibiotics", "IV fluids", "Pain management"]
    },
    "Glaucoma": {
        title: "Glaucoma",
        causes: "Damage to the optic nerve due to high eye pressure.",
        symptoms: ["Blurred vision", "Severe eye pain", "Tunnel vision", "Headaches"],
        stages: [
            "Stage 1: No noticeable symptoms, slight eye pressure increase",
            "Stage 2: Peripheral vision loss begins",
            "Stage 3: Severe vision impairment, potential blindness"
        ],
        emergencyMedicine: ["Eye drops to reduce pressure", "Laser surgery", "Medication"]
    },
    "Glioma": {
        title: "Glioma",
        causes: "A type of tumor that occurs in the brain or spinal cord.",
        symptoms: ["Headaches", "Seizures", "Memory loss", "Personality changes"],
        stages: [
            "Stage 1: Mild symptoms like occasional headaches",
            "Stage 2: Tumor growth causes increased pressure in the brain",
            "Stage 3: Severe neurological damage, loss of cognitive functions"
        ],
        emergencyMedicine: ["Surgery", "Radiation therapy", "Chemotherapy"]
    },
    "Meningioma": {
        title: "Meningioma",
        causes: "A tumor that forms on membranes covering the brain and spinal cord.",
        symptoms: ["Headaches", "Seizures", "Weakness in limbs", "Vision problems"],
        stages: [
            "Stage 1: Small tumor with mild symptoms",
            "Stage 2: Tumor grows, causing neurological issues",
            "Stage 3: Severe brain pressure, possible paralysis"
        ],
        emergencyMedicine: ["Surgery", "Radiotherapy", "Medication for swelling"]
    },
    "Pituitary": {
        title: "Pituitary Tumor",
        causes: "Abnormal growth in the pituitary gland affecting hormone production.",
        symptoms: ["Hormonal imbalances", "Vision problems", "Unexplained weight changes", "Fatigue"],
        stages: [
            "Stage 1: Small tumor causing mild hormonal changes",
            "Stage 2: Tumor affects hormone regulation, causing severe symptoms",
            "Stage 3: Large tumor pressing on optic nerves, leading to vision loss"
        ],
        emergencyMedicine: ["Hormone therapy", "Surgery", "Radiation therapy"]
    },
    "Bone Fracture": {
        title: "Bone Fracture",
        causes: "A break or crack in the bone, usually caused by trauma, overuse, or conditions like osteoporosis.",
        symptoms: ["Pain at the fracture site", "Swelling and bruising", "Deformity in the affected area", "Limited movement or function"],
        stages: [
            "Stage 1: Hairline or small fracture with minimal swelling",
            "Stage 2: Fracture with noticeable pain and swelling, may require immobilization",
            "Stage 3: Severe fracture with bone displacement, requiring surgery or complex treatments"
        ],
        emergencyMedicine: ["Pain management", "Casting or splinting", "Surgical intervention", "Physical therapy for rehabilitation"]
    }
};

// Function to Show Disease Pop-up
function showPopup(diseaseName) {
    if (diseaseData[diseaseName]) {
        const disease = diseaseData[diseaseName];

        let content = `<h3>${disease.title}</h3>`;
        content += `<p><strong>Causes:</strong> ${disease.causes}</p>`;
        content += `<p><strong>Symptoms:</strong></p><ul>`;
        disease.symptoms.forEach(symptom => {
            content += `<li>${symptom}</li>`;
        });
        content += `</ul><p><strong>Stages:</strong></p><ul>`;
        disease.stages.forEach(stage => {
            content += `<li>${stage}</li>`;
        });
        content += `</ul><p><strong>Emergency Medicine:</strong></p><ul>`;
        disease.emergencyMedicine.forEach(medicine => {
            content += `<li>${medicine}</li>`;
        });
        content += `</ul>`;

        document.getElementById("popupContent").innerHTML = content;
        document.getElementById("diseasePopup").style.display = "flex";
    }
}

// Function to Close Pop-up
function closePopup() {
    document.getElementById("diseasePopup").style.display = "none";
}

// Function to Toggle Disease Section Visibility
function toggleDiseaseSection() {
    const section = document.getElementById("diseaseListSection");
    if (section.style.display === "none" || section.style.display === "") {
        section.style.display = "block";
    } else {
        section.style.display = "none";
    }
}


