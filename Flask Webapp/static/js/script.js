// ✅ Toggle Menu Function
function toggleMenu() {
    var menu = document.getElementById("dropdownMenu");
    menu.style.display = menu.style.display === "block" ? "none" : "block";
}

// ✅ Read More Function - Redirect to another page
function readMore() {
    window.location.href = "/about";
}
// Extra 1
const testimonials = [
    { text: "This AI-powered diagnosis is a game changer!", author: "Dr. A. Sharma" },
    { text: "The accuracy of the diagnosis is outstanding.", author: "Dr. R. Verma" },
    { text: "A must-have tool for healthcare professionals.", author: "Dr. S. Gupta" }
];

let currentIndex = 0;

function updateTestimonial() {
    document.getElementById("testimonial-text").innerText = `"${testimonials[currentIndex].text}"`;
    document.getElementById("testimonial-author").innerText = `- ${testimonials[currentIndex].author}`;
}

function prevTestimonial() {
    currentIndex = (currentIndex - 1 + testimonials.length) % testimonials.length;
    updateTestimonial();
}

function nextTestimonial() {
    currentIndex = (currentIndex + 1) % testimonials.length;
    updateTestimonial();
}

// Auto-slide every 4 seconds
setInterval(nextTestimonial, 4000);

// Extra 3
// ✅ Live Statistics - Real-time Updates
let diagnosesCompleted = 1200;
let diseasesDetected = 50;

function updateStats() {
    diagnosesCompleted += Math.floor(Math.random() * 5);
    diseasesDetected += Math.floor(Math.random() * 2);
    
    document.getElementById("diagnoses-count").innerText = diagnosesCompleted;
    document.getElementById("diseases-detected").innerText = diseasesDetected;
}

// Update every 3 seconds
setInterval(updateStats, 3000);

// ✅ Image Carousel (Auto Sliding)
document.addEventListener("DOMContentLoaded", function () {
    let index = 0;
    const images = document.querySelectorAll(".carousel-inner img");
    const totalImages = images.length;
    const carouselInner = document.querySelector(".carousel-inner");

    function nextSlide() {
        index = (index + 1) % totalImages;
        carouselInner.style.transition = "transform 0.8s ease-in-out";
        carouselInner.style.transform = `translateX(-${index * 100}%)`;

        // **Seamless Loop Fix**
        if (index === totalImages - 1) {
            setTimeout(() => {
                carouselInner.style.transition = "none";
                index = 0;
                carouselInner.style.transform = "translateX(0)";
            }, 800); // Wait for animation to complete
        }
    }

    setInterval(nextSlide, 3000); // Change image every 3 seconds
});
