# CANCER_PREDICTION_MODEL
## Breast Cancer Prediction using Random Forest

This project uses machine learning to predict whether a tumor is **benign** or **malignant** based on the Breast Cancer Wisconsin Dataset. The analysis includes data preprocessing, exploratory data analysis (EDA), and building a Random Forest model for prediction.

## Dataset
The dataset contains 32 columns including:
- `id`: Unique identifier for each patient
- `diagnosis`: Target variable (M = Malignant, B = Benign)
- 30 numerical features describing cell properties like radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension.

Columns `id` and `Unnamed: 32` are dropped as they are not useful for modeling.

## Data Preprocessing
- Encode categorical data (`diagnosis`) to numeric (0 = Benign, 1 = Malignant)
- Split dataset into features (X) and target (y)
- Train-test split: 80% training, 20% testing

## Exploratory Data Analysis (EDA)
- Distribution of benign vs malignant tumors
- Correlation heatmap to see relationships between features
- Histograms of numerical features
- Pairplot of first 5 features to visualize class separation
- Feature importance to identify significant features

## Model
- **Algorithm:** Random Forest Classifier
- **Training:** 80% of dataset
- **Testing:** 20% of dataset
- **Evaluation Metric:** Accuracy

**Example Result:**
