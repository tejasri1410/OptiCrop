/* ====================================================================================
   FILE    : script.js
   PROJECT : OptiCrop - Smart Agricultural Production Optimization Engine
   PURPOSE : Adds small client-side interactivity to improve the user experience:
               1. Auto-hides flash messages after a few seconds
               2. Adds a subtle "fade + rise" entrance animation to cards as they
                  scroll into view (using IntersectionObserver)
               3. Provides real-time visual feedback (green/red border) as the
                  user types numbers into the form fields, so they know
                  immediately if a value looks valid -- BEFORE they even submit.

   NOTE FOR BEGINNERS:
   This file does NOT do any Machine Learning. All ML predictions happen on the
   BACKEND (app.py + model/train_model.py). This file only makes the already-
   working website feel smoother and more responsive to the user.
==================================================================================== */

document.addEventListener("DOMContentLoaded", function () {

    // --------------------------------------------------------------------------
    // FEATURE 1: Auto-dismiss flash messages after 5 seconds
    // --------------------------------------------------------------------------
    // Flash messages (like "Please provide a value for Nitrogen") are shown at
    // the top of the page. We don't want them to sit there forever, so we fade
    // them out automatically after a short delay.
    const flashMessages = document.querySelectorAll(".flash-message");
    flashMessages.forEach(function (msg) {
        setTimeout(function () {
            msg.style.transition = "opacity 0.6s ease";
            msg.style.opacity = "0";
            setTimeout(function () { msg.remove(); }, 600);
        }, 5000);
    });

    // --------------------------------------------------------------------------
    // FEATURE 2: Scroll-reveal animation for cards
    // --------------------------------------------------------------------------
    // IntersectionObserver watches elements and tells us when they enter the
    // visible part of the screen ("viewport"). When a card becomes visible,
    // we add a CSS class that triggers a gentle fade+rise animation.
    const revealTargets = document.querySelectorAll(
        ".feature-card, .step-item, .rank-card, .chart-card, .about-card"
    );

    if ("IntersectionObserver" in window && revealTargets.length > 0) {
        const observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = "1";
                    entry.target.style.transform = "translateY(0)";
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.15 });

        revealTargets.forEach(function (el) {
            // Start each card slightly lower & transparent, then animate in
            el.style.opacity = "0";
            el.style.transform = "translateY(18px)";
            el.style.transition = "opacity 0.5s ease, transform 0.5s ease";
            observer.observe(el);
        });
    }

    // --------------------------------------------------------------------------
    // FEATURE 3: Real-time numeric input validation feedback
    // --------------------------------------------------------------------------
    // As the farmer types into a number field (like Nitrogen or pH), we give
    // instant visual feedback: a green border if the number looks realistic,
    // or an amber border if it looks out of the expected range. This helps
    // catch typos BEFORE the form is even submitted to the server.
    const REALISTIC_RANGES = {
        N: [0, 300], P: [0, 300], K: [0, 300],
        temperature: [-10, 60], humidity: [0, 100],
        ph: [0, 14], rainfall: [0, 600]
    };

    Object.keys(REALISTIC_RANGES).forEach(function (fieldName) {
        const input = document.getElementById(fieldName);
        if (!input) return; // Field might not exist on every page

        input.addEventListener("input", function () {
            const value = parseFloat(input.value);
            const [min, max] = REALISTIC_RANGES[fieldName];

            if (input.value.trim() === "") {
                input.style.borderColor = ""; // Reset to default
                return;
            }

            if (isNaN(value) || value < min || value > max) {
                input.style.borderColor = "#B9822B"; // Amber warning color
            } else {
                input.style.borderColor = "#2F5D3A"; // Confirmed green
            }
        });
    });

    // --------------------------------------------------------------------------
    // FEATURE 4: Animate confidence bars on page load (result pages)
    // --------------------------------------------------------------------------
    // The confidence bar width is set inline via Jinja2 (from the backend), but
    // we briefly reset it to 0 and animate it growing to its real value, which
    // gives a satisfying "filling up" visual effect when the result page loads.
    const confidenceBars = document.querySelectorAll(".confidence-bar-fill");
    confidenceBars.forEach(function (bar) {
        const targetWidth = bar.style.width;
        bar.style.width = "0%";
        setTimeout(function () {
            bar.style.width = targetWidth;
        }, 200);
    });

});
