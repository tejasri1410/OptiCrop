"""
====================================================================================
 FILE        : app.py
 PROJECT     : OptiCrop - Smart Agricultural Production Optimization Engine
 PURPOSE     : This is the MAIN ENTRY POINT of the entire web application.
               It is the "bridge" that connects:
                    Frontend (HTML/CSS/JS pages the farmer sees)
                            <----->
                    Backend Machine Learning Model (crop_model.pkl)

 HOW THE FLOW WORKS (Beginner Explanation):
 -------------------------------------------
 1. Flask starts a small web server on your computer (or any server).
 2. When a user visits a URL (called a "route", e.g. "/predict"), Flask
    runs the matching Python function below and returns an HTML page
    (or JSON data) back to the user's browser.
 3. When a farmer submits the crop-recommendation form on the website,
    the browser sends that data to our "/predict" route via an HTTP
    POST request. This file receives that data, feeds it into our
    pre-trained ML model, and sends the predicted crop name back to
    be displayed beautifully on the results page.

 ROUTES (URLs) DEFINED IN THIS FILE:
 -------------------------------------
   GET  "/"                -> Home page (introduction to OptiCrop)
   GET  "/recommend"       -> Scenario 1: Crop recommendation input form
   POST "/predict"         -> Scenario 1: Handles form submission & returns prediction
   GET  "/suitability"     -> Scenario 2: Crop suitability checker input form
   POST "/check_suitability"-> Scenario 2: Handles suitability check & returns result
   GET  "/research"        -> Scenario 3: Research & Analytics Dashboard (EDA charts)
   GET  "/about"           -> About page (project details, tech stack)
   POST "/api/predict"     -> JSON REST API endpoint (for developers/researchers)
====================================================================================
"""

# ------------------------------------------------------------------------------
# STEP 1: Import required libraries
# ------------------------------------------------------------------------------
import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, flash

# ------------------------------------------------------------------------------
# STEP 2: Initialize the Flask application
# ------------------------------------------------------------------------------
# __name__ tells Flask where to look for templates/ and static/ folders.
app = Flask(__name__)
app.secret_key = "opticrop_secret_key_2026"  # Needed for flash messages (form validation feedback)

# ------------------------------------------------------------------------------
# STEP 3: Define important paths & constants
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance.pkl")

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Human-friendly ideal growing condition ranges per crop (used for Scenario 2:
# Suitability checking). These mirror the ranges used to generate our dataset,
# and let us give the user clear, explainable feedback like:
#   "Your Nitrogen level (45) is BELOW the ideal range (80-120) for Rice."
IDEAL_RANGES = {
    "rice":        {"N": (80, 120),  "P": (40, 60),  "K": (35, 45),  "temperature": (20, 27), "humidity": (80, 90), "ph": (5.0, 6.5), "rainfall": (180, 300)},
    "maize":       {"N": (60, 100),  "P": (35, 60),  "K": (15, 25),  "temperature": (18, 27), "humidity": (55, 75), "ph": (5.5, 7.0), "rainfall": (60, 110)},
    "chickpea":    {"N": (20, 60),   "P": (55, 80),  "K": (75, 100), "temperature": (17, 21), "humidity": (14, 20), "ph": (5.5, 7.0), "rainfall": (65, 95)},
    "kidneybeans": {"N": (15, 40),   "P": (55, 80),  "K": (15, 25),  "temperature": (15, 22), "humidity": (18, 24), "ph": (5.5, 6.0), "rainfall": (60, 110)},
    "pigeonpeas":  {"N": (10, 40),   "P": (55, 80),  "K": (15, 25),  "temperature": (18, 37), "humidity": (30, 70), "ph": (4.5, 7.0), "rainfall": (90, 210)},
    "mothbeans":   {"N": (10, 40),   "P": (35, 60),  "K": (15, 25),  "temperature": (24, 32), "humidity": (30, 60), "ph": (3.5, 10.0),"rainfall": (35, 65)},
    "mungbean":    {"N": (10, 40),   "P": (35, 60),  "K": (15, 25),  "temperature": (27, 32), "humidity": (75, 90), "ph": (6.0, 7.5), "rainfall": (45, 65)},
    "blackgram":   {"N": (20, 60),   "P": (55, 80),  "K": (15, 25),  "temperature": (25, 35), "humidity": (60, 70), "ph": (6.0, 7.5), "rainfall": (55, 75)},
    "lentil":      {"N": (10, 40),   "P": (55, 80),  "K": (15, 25),  "temperature": (18, 30), "humidity": (60, 70), "ph": (5.5, 7.0), "rainfall": (35, 55)},
    "pomegranate": {"N": (10, 40),   "P": (10, 40),  "K": (35, 50),  "temperature": (18, 25), "humidity": (85, 95), "ph": (5.5, 7.0), "rainfall": (100, 120)},
    "banana":      {"N": (80, 120),  "P": (70, 95),  "K": (45, 55),  "temperature": (23, 30), "humidity": (75, 85), "ph": (5.5, 6.5), "rainfall": (95, 115)},
    "mango":       {"N": (10, 40),   "P": (15, 40),  "K": (25, 45),  "temperature": (27, 37), "humidity": (45, 55), "ph": (4.5, 7.0), "rainfall": (85, 105)},
    "grapes":      {"N": (10, 40),   "P": (110,145), "K": (190,205), "temperature": (8, 42),  "humidity": (80, 84), "ph": (5.5, 6.5), "rainfall": (60, 75)},
    "watermelon":  {"N": (80, 120),  "P": (10, 40),  "K": (45, 55),  "temperature": (24, 27), "humidity": (80, 90), "ph": (6.0, 7.0), "rainfall": (40, 55)},
    "muskmelon":   {"N": (80, 120),  "P": (10, 40),  "K": (45, 55),  "temperature": (27, 30), "humidity": (90, 95), "ph": (6.0, 7.0), "rainfall": (20, 30)},
    "apple":       {"N": (0, 40),    "P": (110,145), "K": (190,205), "temperature": (21, 24), "humidity": (90, 95), "ph": (5.5, 6.5), "rainfall": (100, 125)},
    "orange":      {"N": (10, 40),   "P": (5, 30),   "K": (5, 20),   "temperature": (10, 35), "humidity": (90, 95), "ph": (6.0, 8.0), "rainfall": (100, 120)},
    "papaya":      {"N": (30, 70),   "P": (45, 70),  "K": (45, 55),  "temperature": (23, 44), "humidity": (90, 95), "ph": (6.5, 7.0), "rainfall": (40, 250)},
    "coconut":     {"N": (10, 40),   "P": (5, 30),   "K": (25, 35),  "temperature": (25, 30), "humidity": (90, 100),"ph": (5.5, 6.5), "rainfall": (140, 230)},
    "cotton":      {"N": (100,140),  "P": (35, 60),  "K": (15, 25),  "temperature": (22, 26), "humidity": (75, 85), "ph": (5.5, 7.0), "rainfall": (60, 100)},
    "jute":        {"N": (60, 100),  "P": (35, 60),  "K": (35, 45),  "temperature": (23, 27), "humidity": (70, 90), "ph": (5.5, 6.5), "rainfall": (150, 200)},
    "coffee":      {"N": (80, 120),  "P": (15, 40),  "K": (25, 35),  "temperature": (23, 28), "humidity": (50, 70), "ph": (6.0, 7.5), "rainfall": (150, 200)},
}

FEATURE_LABELS = {
    "N": "Nitrogen (N)", "P": "Phosphorous (P)", "K": "Potassium (K)",
    "temperature": "Temperature (°C)", "humidity": "Humidity (%)",
    "ph": "Soil pH", "rainfall": "Rainfall (mm)"
}

# ------------------------------------------------------------------------------
# STEP 4: Load the pre-trained Machine Learning artifacts ONCE at server startup
# ------------------------------------------------------------------------------
# We load these ONE TIME when the Flask app starts (not on every request) so
# predictions are extremely fast. If the model files don't exist yet, we show
# a clear, friendly error instead of crashing confusingly.
try:
    crop_model = joblib.load(MODEL_PATH)
    feature_scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    feature_importance = joblib.load(FEATURE_IMPORTANCE_PATH)
    MODEL_LOADED = True
    print("[OptiCrop] Machine Learning model, scaler & encoder loaded successfully.")
except FileNotFoundError:
    MODEL_LOADED = False
    print("[OptiCrop] WARNING: Model files not found. Please run 'python model/train_model.py' first.")


def get_top_predictions(input_df, top_n=3):
    """
    ------------------------------------------------------------------
    FUNCTION : get_top_predictions
    PURPOSE  : Instead of only returning the single "best" crop, this
               function returns the TOP N most suitable crops along
               with their confidence percentage. This gives farmers
               more flexible options to choose from (Scenario 1).

    PARAMETERS:
      input_df (DataFrame) : a single-row DataFrame with the 7 scaled
                              input features, in the exact column order
                              the model was trained on.
      top_n (int) : how many top crop suggestions to return (default 3)

    RETURNS:
      A list of dictionaries: [{"crop": "rice", "confidence": 92.3}, ...]
      sorted from highest to lowest confidence.
    ------------------------------------------------------------------
    """
    # predict_proba() returns the probability of EVERY possible crop class
    probabilities = crop_model.predict_proba(input_df)[0]

    # Pair each crop name with its predicted probability
    crop_names = label_encoder.classes_
    paired = list(zip(crop_names, probabilities))

    # Sort by probability (highest confidence first) and take the top N
    paired_sorted = sorted(paired, key=lambda x: x[1], reverse=True)[:top_n]

    results = [
        {"crop": crop.capitalize(), "confidence": round(float(prob) * 100, 2)}
        for crop, prob in paired_sorted
    ]
    return results


def validate_and_extract_inputs(form_data):
    """
    ------------------------------------------------------------------
    FUNCTION : validate_and_extract_inputs
    PURPOSE  : Safely extracts and validates the 7 numeric inputs coming
               from an HTML form. Prevents the app from crashing if a
               user submits empty, non-numeric, or out-of-range values.

    PARAMETERS:
      form_data : the request.form object from Flask (form submission)

    RETURNS:
      A tuple: (values_dict_or_None, error_message_or_None)
      - If validation succeeds: (dict_of_float_values, None)
      - If validation fails:    (None, "readable error message")
    ------------------------------------------------------------------
    """
    values = {}
    for feature in FEATURE_COLUMNS:
        raw_value = form_data.get(feature, "").strip()
        if raw_value == "":
            return None, f"Please provide a value for {FEATURE_LABELS[feature]}."
        try:
            values[feature] = float(raw_value)
        except ValueError:
            return None, f"{FEATURE_LABELS[feature]} must be a valid number."

    # Basic sanity range checks (soft real-world bounds) to catch typos
    sanity_bounds = {
        "N": (0, 300), "P": (0, 300), "K": (0, 300),
        "temperature": (-10, 60), "humidity": (0, 100),
        "ph": (0, 14), "rainfall": (0, 600)
    }
    for feature, (low, high) in sanity_bounds.items():
        if not (low <= values[feature] <= high):
            return None, (f"{FEATURE_LABELS[feature]} value ({values[feature]}) looks unrealistic. "
                           f"Expected a value between {low} and {high}.")

    return values, None


# ==============================================================================
# ROUTE DEFINITIONS
# ==============================================================================

@app.route("/")
def home():
    """
    Home page -> gives an overview of OptiCrop and links to all 3 scenarios.
    """
    return render_template("index.html", model_loaded=MODEL_LOADED)


@app.route("/recommend", methods=["GET"])
def recommend_form():
    """
    Scenario 1: Displays the empty crop-recommendation input form
    where a farmer enters N, P, K, temperature, humidity, ph, rainfall.
    """
    return render_template("recommend.html", model_loaded=MODEL_LOADED)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Scenario 1 (continued): Handles the submitted form data, runs it
    through the trained ML model, and renders the result page showing
    the top recommended crop(s) with confidence scores.
    """
    if not MODEL_LOADED:
        flash("Model is not trained yet. Please run 'python model/train_model.py' first.", "error")
        return render_template("recommend.html", model_loaded=MODEL_LOADED)

    # Validate and extract the 7 inputs submitted from the HTML form
    values, error = validate_and_extract_inputs(request.form)
    if error:
        flash(error, "error")
        return render_template("recommend.html", model_loaded=MODEL_LOADED, form_values=request.form)

    # Build a single-row DataFrame in the EXACT SAME column order used during training
    input_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)

    # Apply the SAME scaler that was fitted during training (very important:
    # never fit a new scaler here, always reuse the one saved during training)
    scaled_input = feature_scaler.transform(input_df)

    # Get the top 3 most suitable crops with confidence percentages
    top_crops = get_top_predictions(scaled_input, top_n=3)
    best_crop = top_crops[0]

    return render_template(
        "result.html",
        model_loaded=MODEL_LOADED,
        best_crop=best_crop,
        top_crops=top_crops,
        user_inputs=values,
        feature_labels=FEATURE_LABELS
    )


@app.route("/suitability", methods=["GET"])
def suitability_form():
    """
    Scenario 2: Displays the crop suitability checker form, where the
    user picks a SPECIFIC crop they already have in mind and enters
    their soil/climate data to check if it's a good match.
    """
    crop_list = sorted(IDEAL_RANGES.keys())
    return render_template(
        "suitability.html",
        model_loaded=MODEL_LOADED,
        crop_list=crop_list,
        form_values=None,
        ideal_ranges=IDEAL_RANGES,
        selected_crop=None
    )


@app.route("/check_suitability", methods=["POST"])
def check_suitability():
    """
    Scenario 2 (continued): Compares the user's soil/climate values
    against the ideal agronomic range for their CHOSEN crop, and gives
    a clear, factor-by-factor suitability breakdown plus an overall
    suitability score (%).
    """
    selected_crop = request.form.get("crop", "").strip().lower()
    if selected_crop not in IDEAL_RANGES:
        flash("Please select a valid crop from the list.", "error")
        return render_template(
            "suitability.html",
            model_loaded=MODEL_LOADED,
            crop_list=sorted(IDEAL_RANGES.keys()),
            form_values=request.form,
            ideal_ranges=IDEAL_RANGES,
            selected_crop=None
        )

    values, error = validate_and_extract_inputs(request.form)
    if error:
        flash(error, "error")
        crop_list = sorted(IDEAL_RANGES.keys())
        return render_template(
            "suitability.html",
            model_loaded=MODEL_LOADED,
            crop_list=crop_list,
            form_values=request.form,
            selected_crop=selected_crop,
            ideal_ranges=IDEAL_RANGES
        )

    ideal = IDEAL_RANGES[selected_crop]
    breakdown = []
    matched_count = 0

    # Compare each of the 7 features against the ideal range for the chosen crop
    for feature in FEATURE_COLUMNS:
        low, high = ideal[feature]
        user_value = values[feature]
        is_suitable = low <= user_value <= high

        if is_suitable:
            matched_count += 1
            status = "Suitable"
        elif user_value < low:
            status = "Below Ideal Range"
        else:
            status = "Above Ideal Range"

        breakdown.append({
            "feature": FEATURE_LABELS[feature],
            "user_value": user_value,
            "ideal_range": f"{low} - {high}",
            "status": status,
            "is_suitable": is_suitable
        })

    # Overall suitability score = % of the 7 factors that fall within ideal range
    suitability_score = round((matched_count / len(FEATURE_COLUMNS)) * 100, 1)

    # Also run the SAME data through our trained ML model to see what crop
    # it WOULD have recommended -- useful cross-check for the user
    ml_suggestion = None
    if MODEL_LOADED:
        input_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        scaled_input = feature_scaler.transform(input_df)
        ml_suggestion = get_top_predictions(scaled_input, top_n=1)[0]

    return render_template(
        "suitability_result.html",
        model_loaded=MODEL_LOADED,
        selected_crop=selected_crop.capitalize(),
        breakdown=breakdown,
        suitability_score=suitability_score,
        ml_suggestion=ml_suggestion
    )


@app.route("/research")
def research_dashboard():
    """
    Scenario 3: Research & Policy Planning Dashboard. Displays the EDA
    charts (correlation heatmap, feature distributions, confusion matrix,
    feature importance) generated during training, along with the model's
    accuracy report -- giving researchers/policymakers data-driven insight
    into crop-environment relationships.
    """
    accuracy_report = "Model not trained yet."
    report_path = os.path.join(MODEL_DIR, "classification_report.txt")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            accuracy_report = f.read()

    # Sort feature importance for a clean descending display
    sorted_importance = []
    if MODEL_LOADED:
        sorted_importance = sorted(
            [{"feature": FEATURE_LABELS[k], "importance": round(v * 100, 2)} for k, v in feature_importance.items()],
            key=lambda x: x["importance"], reverse=True
        )

    return render_template(
        "research.html",
        model_loaded=MODEL_LOADED,
        accuracy_report=accuracy_report,
        feature_importance=sorted_importance,
        total_crops=len(IDEAL_RANGES)
    )


@app.route("/about")
def about():
    """
    About page -> explains the OptiCrop project, its purpose, the tech
    stack used, and the three real-world use case scenarios it solves.
    """
    return render_template("about.html", model_loaded=MODEL_LOADED)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    ------------------------------------------------------------------
    JSON REST API ENDPOINT (for developers / other systems / researchers)
    ------------------------------------------------------------------
    Allows external systems (e.g., a mobile app, IoT sensor dashboard, or
    another researcher's script) to send a JSON payload and get a JSON
    prediction back -- without needing to load the HTML web pages at all.

    EXAMPLE REQUEST (using curl):
        curl -X POST http://127.0.0.1:5000/api/predict \\
             -H "Content-Type: application/json" \\
             -d '{"N":90,"P":42,"K":43,"temperature":20.8,
                  "humidity":82,"ph":6.5,"rainfall":202.9}'

    EXAMPLE RESPONSE:
        {
          "success": true,
          "top_recommendations": [
             {"crop": "Rice", "confidence": 97.6},
             {"crop": "Jute", "confidence": 1.8},
             {"crop": "Coconut", "confidence": 0.4}
          ]
        }
    ------------------------------------------------------------------
    """
    if not MODEL_LOADED:
        return jsonify({"success": False, "error": "Model not trained yet."}), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"success": False, "error": "Request must be valid JSON."}), 400

    values, error = validate_and_extract_inputs(
        {k: str(v) for k, v in data.items()} if data else {}
    )
    if error:
        return jsonify({"success": False, "error": error}), 400

    input_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    scaled_input = feature_scaler.transform(input_df)
    top_crops = get_top_predictions(scaled_input, top_n=3)

    return jsonify({"success": True, "top_recommendations": top_crops})


@app.errorhandler(404)
def page_not_found(e):
    """Custom friendly 404 page instead of Flask's default plain error."""
    return render_template("404.html"), 404


# ------------------------------------------------------------------------------
# STEP 5: Run the Flask development server
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # debug=True -> auto-reloads the server on code changes & shows detailed errors
    # (This should be set to False in a real production deployment)
    app.run(debug=True, host="0.0.0.0", port=5000)
