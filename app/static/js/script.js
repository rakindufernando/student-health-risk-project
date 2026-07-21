document.addEventListener("DOMContentLoaded", function () {
    const predictionForm = document.querySelector(".prediction-form");
    const submitButton = document.querySelector(
        ".prediction-form button[type='submit']"
    );
    const resetButton = document.querySelector(
        ".prediction-form button[type='reset']"
    );
    const errorMessage = document.querySelector(".error-message");
    const animatedBars = document.querySelectorAll(".probability-animation");

    function restoreSubmitButton() {
        if (!submitButton) {
            return;
        }

        submitButton.disabled = false;
        submitButton.textContent = "Predict Health Condition";
    }

    if (predictionForm && submitButton) {
        predictionForm.addEventListener("submit", function (event) {
            const invalidField = predictionForm.querySelector(":invalid");

            if (invalidField) {
                event.preventDefault();
                invalidField.classList.add("input-invalid");
                invalidField.focus();
                invalidField.reportValidity();
                return;
            }

            submitButton.disabled = true;
            submitButton.textContent = "Generating Prediction...";
        });

        predictionForm.addEventListener("input", function (event) {
            if (event.target.matches("input, select")) {
                event.target.classList.remove("input-invalid");
            }
        });
    }

    if (predictionForm && resetButton) {
        resetButton.addEventListener("click", function (event) {
            const shouldClear = window.confirm(
                "Do you want to clear all entered values?"
            );

            if (!shouldClear) {
                event.preventDefault();
            }
        });
    }

    if (errorMessage) {
        errorMessage.focus();
        errorMessage.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });
    }

    animatedBars.forEach(function (bar) {
        const finalWidth = bar.dataset.width || "0%";
        bar.style.width = "0%";

        window.setTimeout(function () {
            bar.style.width = finalWidth;
        }, 200);
    });

    window.addEventListener("pageshow", function () {
        restoreSubmitButton();
    });
});
