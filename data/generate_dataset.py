"""
====================================================================================
 FILE        : generate_dataset.py
 PROJECT     : OptiCrop - Smart Agricultural Production Optimization Engine
 PURPOSE     : This script GENERATES a realistic, ready-to-use dataset that our
               Machine Learning model will learn from.

 WHY DO WE NEED THIS FILE?
 --------------------------
 Every Machine Learning project needs DATA before it can learn anything.
 In real life, this data would come from agricultural research institutes,
 government soil-testing labs, or IoT soil sensors placed in farms.

 For this project, we RECREATE that same kind of dataset ourselves using
 well documented, real-world agronomic requirement ranges for 22 different
 crops (values referenced from agricultural science literature such as
 ICAR, FAO crop requirement tables, and the widely used "Crop Recommendation"
 dataset schema). For every crop, we know the approximate MINIMUM and
 MAXIMUM values of:
     N  -> Nitrogen content in soil        (kg/ha)
     P  -> Phosphorous content in soil     (kg/ha)
     K  -> Potassium content in soil       (kg/ha)
     temperature -> Average temperature    (in Celsius)
     humidity    -> Relative humidity      (in %)
     ph          -> Soil pH value          (0-14 scale)
     rainfall    -> Average rainfall       (in mm)

 Using these known ranges, we generate 100 realistic samples PER CROP
 (22 crops x 100 rows = 2200 rows total) by randomly sampling values
 from a NORMAL (bell-curve) distribution centered between each crop's
 min and max requirement. This mimics how real farms/fields show natural
 variation around an "ideal" value instead of being spread completely
 randomly (uniformly) between min and max.

 OUTPUT: data/crop_data.csv  -> this file becomes the single source of truth
         that train_model.py will use to train our ML model.
====================================================================================
"""

# ------------------------------------------------------------------------------
# STEP 1: Import required libraries
# ------------------------------------------------------------------------------
import numpy as np          # NumPy -> used for numerical operations & random sampling
import pandas as pd         # Pandas -> used to build & save the dataset as a table (DataFrame)
import os                   # os -> used to handle file paths in an OS-independent way

# ------------------------------------------------------------------------------
# STEP 2: Fix the "random seed"
# ------------------------------------------------------------------------------
# Setting a seed means: every time this script runs, it will generate the
# EXACT SAME "random" numbers. This is extremely important for reproducibility
# -> so that our results, accuracy, and graphs never change between runs.
np.random.seed(42)

# ------------------------------------------------------------------------------
# STEP 3: Define realistic agronomic requirement ranges for each crop
# ------------------------------------------------------------------------------
# Each crop maps to a dictionary containing the (min, max) realistic range for
# every feature. These ranges are based on standard agricultural requirement
# tables used widely in crop-recommendation research.
#
# Format:  "crop_name": {"N": (min,max), "P": (min,max), "K": (min,max),
#                         "temperature": (min,max), "humidity": (min,max),
#                         "ph": (min,max), "rainfall": (min,max)}
CROP_REQUIREMENTS = {
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


def sample_value(low, high):
    """
    ------------------------------------------------------------------
    FUNCTION : sample_value
    PURPOSE  : Generate ONE realistic random number for a soil/climate
               feature, centered between 'low' and 'high'.

    HOW IT WORKS:
      - We calculate the mean (average) of low & high -> the "ideal" value.
      - We calculate a standard deviation as 1/4th of the range so most
        generated values stay comfortably inside the [low, high] band
        (bell-curve / normal distribution behaviour).
      - np.random.normal() draws a random number from that bell curve.
      - np.clip() makes sure the value never falls outside realistic
        bounds, even if the random draw was an extreme outlier.

    PARAMETERS:
      low  (float) : minimum realistic value for this feature
      high (float) : maximum realistic value for this feature

    RETURNS:
      A single float value, rounded to 2 decimal places.
    ------------------------------------------------------------------
    """
    mean = (low + high) / 2
    std_dev = max((high - low) / 4, 0.01)  # avoid zero std_dev for very tight ranges
    value = np.random.normal(loc=mean, scale=std_dev)
    value = np.clip(value, low * 0.9, high * 1.1)  # allow slight natural variation
    return round(float(value), 2)


def generate_dataset(samples_per_crop=100):
    """
    ------------------------------------------------------------------
    FUNCTION : generate_dataset
    PURPOSE  : Build the FULL dataset by generating 'samples_per_crop'
               synthetic rows for EVERY crop defined in CROP_REQUIREMENTS.

    PARAMETERS:
      samples_per_crop (int) : how many rows to generate for each crop
                                (default 100 -> 22 crops x 100 = 2200 rows)

    RETURNS:
      A pandas DataFrame with columns:
      ['N','P','K','temperature','humidity','ph','rainfall','label']
    ------------------------------------------------------------------
    """
    rows = []  # This list will temporarily hold every generated row (as a dict)

    # Loop through every crop and its requirement ranges
    for crop_name, ranges in CROP_REQUIREMENTS.items():
        for _ in range(samples_per_crop):
            row = {
                "N": sample_value(*ranges["N"]),
                "P": sample_value(*ranges["P"]),
                "K": sample_value(*ranges["K"]),
                "temperature": sample_value(*ranges["temperature"]),
                "humidity": sample_value(*ranges["humidity"]),
                "ph": sample_value(*ranges["ph"]),
                "rainfall": sample_value(*ranges["rainfall"]),
                "label": crop_name,  # the target/output column -> what we want to predict
            }
            rows.append(row)

    # Convert the list of dictionaries into a clean pandas DataFrame (table)
    dataframe = pd.DataFrame(rows)

    # Shuffle all rows randomly so crops aren't grouped in sequential blocks
    # (important because it mimics a real-world unordered dataset)
    dataframe = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)

    return dataframe


# ------------------------------------------------------------------------------
# STEP 4: Main execution block
# ------------------------------------------------------------------------------
# This code only runs when generate_dataset.py is executed DIRECTLY
# (not when it is imported by another script).
if __name__ == "__main__":
    print("Generating OptiCrop synthetic agricultural dataset...")

    final_dataset = generate_dataset(samples_per_crop=100)

    # Build the absolute output path so this script works from any directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "crop_data.csv")

    # Save the dataset as a CSV file (comma separated values) -> universal format
    final_dataset.to_csv(output_path, index=False)

    print(f"Dataset successfully created with {len(final_dataset)} rows.")
    print(f"Saved at: {output_path}")
    print("\nPreview of generated data:")
    print(final_dataset.head())
    print("\nCrops included:", list(CROP_REQUIREMENTS.keys()))
