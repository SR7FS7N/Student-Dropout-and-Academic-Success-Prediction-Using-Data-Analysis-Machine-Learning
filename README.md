# 🎓 Student Dropout & Academic Success Prediction

## 📌 Project Overview

This project analyzes student academic, demographic, and macro-economic data to predict whether a student is likely to **Dropout** or **Graduate**. The project combines **Exploratory Data Analysis (EDA)**, correlation analysis, outlier treatment, machine learning model comparison, and an interactive **Streamlit** web application.

The original dataset contains **4,424 student records and 35 input features**, with three target classes: **Dropout, Enrolled, and Graduate**. There are no missing values or duplicate rows in the supplied dataset.

For the machine learning task, the `Enrolled` class was removed so that the target becomes a binary classification problem:

- `0` → Dropout
- `1` → Graduate

## 📊 Dataset & Data Analysis

The dataset contains information collected at student enrollment, including demographic information, previous qualifications, academic performance, financial information, and macro-economic indicators.

The analysis followed these main steps:

1. **Dataset inspection**
   - Checked dataset shape, data types, unique values, missing values, and duplicate rows.
   - Examined summary statistics and the distribution of the target variable.

2. **Target preprocessing**
   - Removed the `Enrolled` category.
   - Converted `Dropout` to `0` and `Graduate` to `1`.

3. **Correlation analysis**
   - Created a correlation matrix to study relationships between the features and the target.
   - Selected features with a positive correlation with the target for the modelling stage.

4. **Outlier analysis**
   - Used boxplots to inspect the distributions of selected features.
   - Applied the IQR-based outlier filtering method with `k = 10`.
   - The processed dataset contained **2,437 records and 22 selected features** after preprocessing.

5. **Feature scaling and data splitting**
   - Standardized the selected features using `StandardScaler`.
   - Split the data into **67% training** and **33% testing** sets.

## 🤖 Machine Learning Model Selection

Three models were trained and evaluated:

- **Logistic Regression**
- **Gaussian Naive Bayes**
- **Linear Regression** with a `0.5` threshold for converting predictions into binary classes

For Logistic Regression, `GridSearchCV` with 5-fold cross-validation was used to select the best `C` value from:

`[0.001, 0.01, 0.1, 1, 10, 100]`

The best value was **C = 100**.

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9019 | 0.9032 | 0.9019 | 0.8987 |
| Naive Bayes | 0.8857 | 0.8867 | 0.8857 | 0.8815 |
| Linear Regression | 0.9019 | 0.9095 | 0.9019 | 0.8967 |

**Logistic Regression was selected as the final prediction model.**

Although Linear Regression achieved the same accuracy, Logistic Regression is the appropriate classification model for the binary Dropout/Graduate prediction task and also provides class probabilities and interpretable coefficients.

The final Logistic Regression model, scaler, feature names, and target classes were saved as `.pkl` files using `joblib`.

## 🖥️ Streamlit Application (`app.py`)

The `app.py` file converts the analysis and machine learning workflow into an interactive Streamlit application.

The application contains five main sections:

### 1. Overview

Displays:

- Dataset size
- Number of features
- Missing values
- Duplicate rows
- Dataset preview
- Summary statistics
- Column data types
- Missing-value information
- Preprocessing pipeline

### 2. Exploratory Analysis

Provides interactive visual analysis including:

- Target distribution using pie and bar charts
- Gender distribution
- Student outcome by gender
- Student outcome by marital status
- Feature distributions using histograms
- **Boxplots** showing feature values by student outcome

A feature can be selected from the interface to explore its distribution and relationship with the target.

### 3. Correlation Analysis

Shows:

- Feature-to-target correlation
- Correlation heatmaps
- Selected positively correlated features
- Before/after outlier analysis
- Boxplots for the selected features

The application also allows the user to render the complete correlation heatmap.

### 4. Model Training & Comparison

The application trains and compares the three models and displays:

- Accuracy
- Precision
- Recall
- F1-score
- Model comparison charts
- Confusion matrices
- Classification reports
- Logistic Regression feature coefficients

The models are ranked by accuracy so their performance can be compared easily.

### 5. Prediction

The **Predict** page uses the selected Logistic Regression model to predict student outcomes.

For a single student, users can:

- Enter or adjust feature values
- Start with a median student
- Start from a real Graduate or Dropout example
- Generate a prediction

The application returns:

- Predicted outcome
- Prediction confidence
- Probability of Dropout
- Probability of Graduate
- Top factors contributing to the prediction

The app also supports **batch prediction**. Users can upload a CSV containing the required feature columns, generate predictions for multiple students, view the results, and download the prediction output as a CSV file.

If the saved model files are unavailable, `app.py` can train the models during the current session and use the Logistic Regression model for prediction.

## 📁 Project Structure

```text
student-dropout-prediction/
│
├── app.py
├── data.csv
├── Untitled.ipynb
├── README.md
│
└── model/
    ├── logistic_regression_model.pkl
    ├── scaler.pkl
    ├── feature_names.pkl
    └── target_classes.pkl
```

> **Note:** `app.py` expects the dataset to be named `data.csv` and located in the same directory as `app.py`.

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

## 🚀 How to Run the Application Locally

### Step 1 — Clone or download the project

Download the project files to your computer.

### Step 2 — Open the project folder

Open a terminal/Command Prompt inside the project directory.

### Step 3 — Create a virtual environment

```bash
python -m venv venv
```

### Step 4 — Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### Step 5 — Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

### Step 6 — Prepare the dataset

Make sure the dataset is in the same folder as `app.py` and is named:

```text
data.csv
```

If your downloaded file has another name, rename it to `data.csv`.

### Step 7 — Make sure the model folder is available

If you have the saved `.pkl` files, place them inside:

```text
model/
```

The required files are:

```text
logistic_regression_model.pkl
scaler.pkl
feature_names.pkl
```

If these files are not available, the application can train the models during the session.

### Step 8 — Run the Streamlit application

```bash
streamlit run app.py
```

### Step 9 — Open the application

Streamlit will provide a local address, normally:

```text
http://localhost:8501
```

Open this address in your browser.

## 🔄 Application Workflow

```text
Dataset
   ↓
Data Inspection
   ↓
Remove "Enrolled"
   ↓
Convert Target to Binary
   ↓
Correlation Analysis
   ↓
Select Positive-Correlation Features
   ↓
Outlier Detection & Removal
   ↓
Feature Standardization
   ↓
Train/Test Split
   ↓
Train Multiple ML Models
   ↓
Compare Accuracy, Precision, Recall & F1
   ↓
Select Logistic Regression
   ↓
Streamlit Prediction App
   ↓
Single or Batch Prediction
```

## 🎯 Project Goal

The main goal of this project is to demonstrate how student data can be transformed from raw records into actionable predictive insights using **data analysis, visualization, feature selection, machine learning, and an interactive web application**.

The final Streamlit application brings the complete workflow together in one place, allowing users to explore the dataset, understand the analysis, compare machine learning models, and make new student outcome predictions.
