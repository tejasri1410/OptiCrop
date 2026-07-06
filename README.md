#  OptiCrop — Smart Agricultural Production Optimization Engine

An end-to-end **AIML web application** that recommends the most suitable crop for a
piece of land based on its soil nutrients (Nitrogen, Phosphorous, Potassium) and
climate conditions (temperature, humidity, pH, rainfall).

Built with **Python, Flask, NumPy, Pandas, Scikit-learn, SciPy, Matplotlib & Seaborn.**

---

##  Problem Statement Recap

OptiCrop integrates key environmental factors — N, P, K, soil temperature, humidity,
pH, and rainfall — to provide intelligent, data-driven crop recommendations to
farmers, researchers, and policymakers, covering three real-world scenarios:

| # | Scenario | Who uses it | What it does |
|---|----------|-------------|--------------|
| 1 | **Smart Crop Recommendation** | Farmers | Enter soil/climate data → get the best crop to plant |
| 2 | **Crop Suitability & Environmental Assessment** | Farmers/Agronomists | Check if current conditions suit a *specific* crop |
| 3 | **Agricultural Research & Policy Planning** | Researchers/Policymakers | Explore data visualizations & crop-environment patterns |

---

##  Project Structure

```
OptiCrop/
│
├── app.py                     # Flask backend — all routes & prediction logic
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── data/
│   ├── generate_dataset.py    # Generates the synthetic-but-realistic training dataset
│   └── crop_data.csv          # Generated dataset (2,200 rows × 22 crops)
│
├── model/
│   ├── train_model.py         # Full ML pipeline: EDA → preprocessing → training → evaluation
│   ├── crop_model.pkl         # Saved trained RandomForestClassifier (generated)
│   ├── scaler.pkl             # Saved StandardScaler (generated)
│   ├── label_encoder.pkl      # Saved LabelEncoder (generated)
│   ├── feature_importance.pkl # Saved feature importance scores (generated)
│   └── classification_report.txt  # Saved model evaluation report (generated)
│
├── static/
│   ├── css/style.css          # All styling for the web app
│   ├── js/script.js           # Client-side interactivity (validation, animations)
│   └── images/                # Auto-generated EDA charts (heatmaps, distributions, etc.)
│
└── templates/                 # Jinja2 HTML templates
    ├── base.html               # Shared layout (navbar, footer)
    ├── index.html              # Home page
    ├── recommend.html          # Scenario 1: input form
    ├── result.html             # Scenario 1: prediction result
    ├── suitability.html        # Scenario 2: input form
    ├── suitability_result.html # Scenario 2: assessment result
    ├── research.html           # Scenario 3: research dashboard
    ├── about.html               # About the project
    └── 404.html                 # Custom error page
```

---

##  Setup & Run Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset (only needed once, already included as `data/crop_data.csv`)
```bash
python data/generate_dataset.py
```

### 3. Train the Machine Learning model (only needed once, already included as `.pkl` files)
```bash
python model/train_model.py
```
This will print accuracy metrics to the console and save:
- The trained model, scaler & label encoder to `model/`
- EDA & evaluation charts to `static/images/`

### 4. Run the Flask web application
```bash
python app.py
```
Then open your browser at **http://127.0.0.1:5000/**

---

##  How the Machine Learning Works

1. **Dataset**: 2,200 samples across 22 crops, each with 7 features (N, P, K,
   temperature, humidity, ph, rainfall), generated using real agronomic
   requirement ranges for each crop.
2. **EDA**: Correlation heatmaps, feature distributions, boxplots, and class
   balance checks using Pandas, Matplotlib, Seaborn & SciPy (z-score outlier detection).
3. **Preprocessing**: `StandardScaler` normalizes all numeric features;
   `LabelEncoder` converts crop names into numbers.
4. **Model**: A `RandomForestClassifier` (300 trees, max depth 15) is trained
   on 80% of the data.
5. **Evaluation**: Accuracy, per-class precision/recall/F1-score, and a
   confusion matrix are computed on the held-out 20% test set.
6. **Serving**: The trained model + scaler + encoder are saved with `joblib`
   and loaded once when the Flask app starts, so every prediction request is
   instant.

---

##  REST API (for developers)

`POST /api/predict` accepts JSON and returns the top-3 crop recommendations:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"N":90,"P":42,"K":43,"temperature":20.8,"humidity":82,"ph":6.5,"rainfall":202.9}'
```

```json
{
  "success": true,
  "top_recommendations": [
    {"crop": "Rice", "confidence": 97.6},
    {"crop": "Jute", "confidence": 1.8},
    {"crop": "Coconut", "confidence": 0.4}
  ]
}
```

---

##  Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn (RandomForestClassifier, StandardScaler, LabelEncoder)
- **Data Handling**: NumPy, Pandas
- **Statistics**: SciPy (z-score outlier detection)
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3 (custom design system), vanilla JavaScript
- **Model Persistence**: Joblib

---

##  Retraining with Your Own Real Data

If you have a real soil-testing dataset (e.g., with columns `N, P, K, temperature,
humidity, ph, rainfall, label`), simply replace `data/crop_data.csv` with your file
(keeping the same column names) and re-run:
```bash
python model/train_model.py
```
The Flask app will automatically pick up the newly trained model on next restart.
