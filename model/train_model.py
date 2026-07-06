"""
====================================================================================
 FILE        : train_model.py
 PROJECT     : OptiCrop - Smart Agricultural Production Optimization Engine
 PURPOSE     : This is the "brain-building" script of the whole project.

 WHAT DOES THIS SCRIPT DO? (High level overview for beginners)
 ------------------------------------------------------------
 1. LOADS the dataset we generated in data/generate_dataset.py
 2. EXPLORES the data (Exploratory Data Analysis / EDA) and saves useful
    charts (using Matplotlib & Seaborn) so we can visually understand
    relationships between soil/climate features and crops.
 3. PREPROCESSES the data:
      - Splits data into "features" (X) and "target/label" (y)
      - Encodes the crop names (text) into numbers using LabelEncoder
        (ML models only understand numbers, not text)
      - Scales/normalizes numeric features using StandardScaler so that
        features with large ranges (like rainfall) don't unfairly
        dominate features with small ranges (like pH)
      - Splits data into TRAINING data (80%) and TESTING data (20%)
 4. TRAINS a Random Forest Classifier -> chosen because it:
      - Handles non-linear relationships between soil/climate features well
      - Is robust to outliers and doesn't require heavy feature engineering
      - Gives us "feature importance" -> tells us WHICH factor (N, P, K,
        temperature, humidity, ph, rainfall) matters most for prediction
      - Works extremely well for tabular/structured agricultural data
 5. EVALUATES the trained model using Accuracy, Classification Report,
    and a Confusion Matrix heatmap (saved as an image).
 6. SAVES the final trained model, the scaler, and the label encoder to
    disk using 'joblib' so that our Flask backend (app.py) can LOAD them
    instantly without re-training every time the server starts.

 WHY SAVE THE MODEL SEPARATELY FROM THE WEB APP?
 ------------------------------------------------
 Training a model can take time (even if it's fast here). We only want to
 train ONCE, then reuse ("serve") that same trained model for every future
 prediction request coming from the website. This is standard real-world
 ML engineering practice (train once, serve many times).
====================================================================================
"""

# ------------------------------------------------------------------------------
# STEP 1: Import all required libraries
# ------------------------------------------------------------------------------
import os                                    # For handling file paths safely
import numpy as np                           # For numerical operations
import pandas as pd                          # For loading & handling tabular data
import matplotlib
matplotlib.use("Agg")                        # Use non-interactive backend (no GUI needed on server)
import matplotlib.pyplot as plt              # For creating charts/plots
import seaborn as sns                        # For beautiful statistical visualizations
import joblib                                # For saving/loading trained ML models to/from disk

from scipy import stats                      # SciPy -> used here for statistical outlier detection (z-score)
from sklearn.model_selection import train_test_split   # To split data into train/test sets
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For scaling & encoding
from sklearn.ensemble import RandomForestClassifier     # Our chosen ML algorithm
from sklearn.metrics import (
    accuracy_score,        # Measures overall correctness of predictions
    classification_report, # Gives precision, recall, f1-score per crop
    confusion_matrix        # Shows which crops get confused with which
)

# ------------------------------------------------------------------------------
# STEP 2: Define important file paths (kept OS-independent using os.path)
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # OptiCrop/ root folder
DATA_PATH = os.path.join(BASE_DIR, "data", "crop_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")

# Make sure the images folder exists (it should, but this is a safety check)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# File paths where we will SAVE our trained artifacts
MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance.pkl")

# The 7 input features our model uses to make predictions
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"


def load_data():
    """
    ------------------------------------------------------------------
    FUNCTION : load_data
    PURPOSE  : Reads the CSV dataset from disk into a pandas DataFrame.
    RETURNS  : pandas.DataFrame containing the full dataset.
    ------------------------------------------------------------------
    """
    print("STEP 1/6: Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"   -> Dataset loaded successfully with shape: {df.shape}")
    print(f"   -> Columns: {list(df.columns)}")
    return df


def explore_data(df):
    """
    ------------------------------------------------------------------
    FUNCTION : explore_data
    PURPOSE  : Performs Exploratory Data Analysis (EDA) and saves several
               charts as PNG images inside static/images/. These charts
               are later displayed on the "Research & Analytics" page of
               our Flask web app for researchers/policymakers (Scenario 3).

    CHARTS GENERATED:
      1. correlation_heatmap.png -> shows how strongly each numeric
         feature (N, P, K, temperature, humidity, ph, rainfall) relates
         to the others. Built using Seaborn.
      2. feature_distribution.png -> histogram/distribution of every
         feature across the whole dataset, so we can see typical ranges.
      3. crop_count.png -> bar chart confirming every crop has an equal
         number of samples (balanced dataset check).
      4. boxplot_nutrients.png -> boxplots comparing N, P, K spread,
         useful for spotting outliers using visual + statistical (SciPy)
         methods.
    ------------------------------------------------------------------
    """
    print("STEP 2/6: Performing Exploratory Data Analysis (EDA)...")

    sns.set_style("whitegrid")  # Clean, professional-looking plot background

    # ---- Chart 1: Correlation Heatmap ----
    # Correlation shows how two numeric variables move together.
    # +1 => perfectly move together, -1 => perfectly opposite, 0 => no relation.
    plt.figure(figsize=(9, 7))
    correlation_matrix = df[FEATURE_COLUMNS].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Soil & Climate Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "correlation_heatmap.png"), dpi=110)
    plt.close()

    # ---- Chart 2: Feature Distributions ----
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    for idx, col in enumerate(FEATURE_COLUMNS):
        sns.histplot(df[col], kde=True, ax=axes[idx], color="#2e7d32")
        axes[idx].set_title(f"Distribution of {col}")
    # Hide any unused subplot axes (since 7 features don't fill a 3x3=9 grid)
    for idx in range(len(FEATURE_COLUMNS), len(axes)):
        fig.delaxes(axes[idx])
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "feature_distribution.png"), dpi=110)
    plt.close()

    # ---- Chart 3: Crop Sample Count (dataset balance check) ----
    plt.figure(figsize=(12, 6))
    crop_counts = df[TARGET_COLUMN].value_counts()
    sns.barplot(x=crop_counts.index, y=crop_counts.values, palette="viridis")
    plt.xticks(rotation=75)
    plt.ylabel("Number of Samples")
    plt.xlabel("Crop")
    plt.title("Number of Data Samples per Crop (Dataset Balance Check)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "crop_count.png"), dpi=110)
    plt.close()

    # ---- Chart 4: Boxplots for N, P, K (outlier detection via visuals) ----
    plt.figure(figsize=(10, 6))
    df_melted = df[["N", "P", "K"]].melt(var_name="Nutrient", value_name="Value")
    sns.boxplot(data=df_melted, x="Nutrient", y="Value", palette="Set2")
    plt.title("Nutrient (N, P, K) Spread & Outlier Check", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "boxplot_nutrients.png"), dpi=110)
    plt.close()

    # ---- SciPy usage: Statistical outlier detection using Z-Score ----
    # A Z-score tells us "how many standard deviations away from the mean"
    # a data point is. Any |z| > 3 is generally considered a strong outlier.
    z_scores = np.abs(stats.zscore(df[FEATURE_COLUMNS]))
    outlier_count = int((z_scores > 3).sum().sum())
    print(f"   -> EDA charts saved to: {IMAGES_DIR}")
    print(f"   -> Statistical outliers detected (|z-score| > 3): {outlier_count} data points")


def preprocess_data(df):
    """
    ------------------------------------------------------------------
    FUNCTION : preprocess_data
    PURPOSE  : Prepares the raw dataset for Machine Learning by encoding
               the target labels and scaling the numeric input features.

    WHY SCALE THE FEATURES?
      Our features have very different natural ranges:
        - ph ranges roughly between 3.5 - 10
        - rainfall ranges roughly between 20 - 300
      Without scaling, "rainfall" would dominate the model's calculations
      simply because its numbers are bigger -- NOT because it's more
      important. StandardScaler transforms every feature to have a
      mean of 0 and a standard deviation of 1, putting them all on a
      level playing field.

    WHY ENCODE THE LABELS?
      ML models work with numbers internally. LabelEncoder converts
      crop names like "rice", "maize", "coffee" into numbers like
      0, 1, 2 ... behind the scenes (and we can always convert back).

    RETURNS:
      X_train, X_test, y_train, y_test, scaler, label_encoder
    ------------------------------------------------------------------
    """
    print("STEP 3/6: Preprocessing data (encoding + scaling)...")

    # Separate input features (X) from the target/output column (y)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Encode crop names into numeric labels (e.g., "apple" -> 0, "banana" -> 1, ...)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Scale all numeric features to a standard range (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training (80%) and testing (20%) sets.
    # 'stratify=y_encoded' ensures every crop is proportionally represented
    # in both the training and testing sets (very important for fairness).
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"   -> Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, scaler, label_encoder


def train_model(X_train, y_train):
    """
    ------------------------------------------------------------------
    FUNCTION : train_model
    PURPOSE  : Trains a Random Forest Classifier on the training data.

    WHAT IS A RANDOM FOREST?
      Imagine asking 300 different "expert" decision trees to each
      independently guess the best crop, and then taking a majority
      vote across all of them. That's essentially a Random Forest.
      Using many trees (instead of just one) makes the final prediction
      far more accurate and stable, and much less likely to "overfit"
      (memorize the training data instead of learning general patterns).

    PARAMETERS EXPLAINED:
      n_estimators=300   -> build 300 individual decision trees
      max_depth=15       -> limit how deep each tree can grow (prevents overfitting)
      random_state=42    -> ensures reproducible results every run
      n_jobs=-1          -> use ALL available CPU cores to train faster
    ------------------------------------------------------------------
    """
    print("STEP 4/6: Training Random Forest Classifier model...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("   -> Model training complete.")
    return model


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    ------------------------------------------------------------------
    FUNCTION : evaluate_model
    PURPOSE  : Checks HOW GOOD our trained model actually is by testing
               it on data it has NEVER seen before (the test set).

    METRICS EXPLAINED:
      - Accuracy: % of total predictions that were correct.
      - Classification Report: Precision, Recall, F1-score for EACH crop
        individually (helps spot if any specific crop is harder to predict).
      - Confusion Matrix: A grid showing actual vs predicted crops, so we
        can see exactly which crops the model sometimes confuses.
    ------------------------------------------------------------------
    """
    print("STEP 5/6: Evaluating model performance on unseen test data...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"   -> Model Accuracy: {accuracy * 100:.2f}%")

    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print("   -> Classification Report:\n", report)

    # Save the classification report to a text file for later reference/display
    with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write(report)

    # ---- Confusion Matrix Heatmap ----
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(13, 11))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.xlabel("Predicted Crop")
    plt.ylabel("Actual Crop")
    plt.title(f"Confusion Matrix (Overall Accuracy: {accuracy*100:.2f}%)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=75)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "confusion_matrix.png"), dpi=110)
    plt.close()

    return accuracy


def save_feature_importance(model):
    """
    ------------------------------------------------------------------
    FUNCTION : save_feature_importance
    PURPOSE  : Random Forest models can tell us WHICH input feature
               (N, P, K, temperature, humidity, ph, rainfall) had the
               biggest influence on predictions overall. This is
               extremely valuable for agricultural researchers
               (Scenario 3) who want to understand crop-environment
               relationships.
    ------------------------------------------------------------------
    """
    importances = model.feature_importances_
    importance_dict = dict(zip(FEATURE_COLUMNS, importances))

    # Save as a .pkl so the Flask app can load and display it later
    joblib.dump(importance_dict, FEATURE_IMPORTANCE_PATH)

    # Also create a nice bar chart of feature importance
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    features_sorted, values_sorted = zip(*sorted_items)

    plt.figure(figsize=(9, 6))
    sns.barplot(x=list(values_sorted), y=list(features_sorted), palette="crest")
    plt.xlabel("Relative Importance")
    plt.title("Which Factors Matter Most for Crop Prediction?", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "feature_importance.png"), dpi=110)
    plt.close()

    print("   -> Feature importance chart & data saved.")
    return importance_dict


def save_artifacts(model, scaler, label_encoder):
    """
    ------------------------------------------------------------------
    FUNCTION : save_artifacts
    PURPOSE  : Persist (save) the trained model, scaler, and label
               encoder to disk using joblib, so the Flask backend
               (app.py) can simply LOAD these files instantly instead
               of re-training the model every time the server restarts.
    ------------------------------------------------------------------
    """
    print("STEP 6/6: Saving trained model, scaler, and label encoder to disk...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    print(f"   -> Model saved at: {MODEL_PATH}")
    print(f"   -> Scaler saved at: {SCALER_PATH}")
    print(f"   -> Label Encoder saved at: {ENCODER_PATH}")


# ------------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# ------------------------------------------------------------------------------
# This runs the full pipeline end-to-end, in order, when you execute:
#     python model/train_model.py
if __name__ == "__main__":
    print("=" * 70)
    print(" OptiCrop - Model Training Pipeline Started")
    print("=" * 70)

    dataset = load_data()
    explore_data(dataset)
    X_train, X_test, y_train, y_test, fitted_scaler, fitted_encoder = preprocess_data(dataset)
    trained_model = train_model(X_train, y_train)
    final_accuracy = evaluate_model(trained_model, X_test, y_test, fitted_encoder)
    save_feature_importance(trained_model)
    save_artifacts(trained_model, fitted_scaler, fitted_encoder)

    print("=" * 70)
    print(f" TRAINING PIPELINE COMPLETE | Final Test Accuracy: {final_accuracy*100:.2f}%")
    print("=" * 70)
