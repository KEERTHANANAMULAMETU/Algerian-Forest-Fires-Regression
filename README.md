
# **Algerian Forest Fires Prediction**

This project predicts the **Fire Weather Index (FWI)** using historical weather and FWI component data. The prediction of FWI is essential for forest fire risk assessment and mitigation strategies.

---

## **Table of Contents**

1. [Introduction](#introduction)  
2. [Problem Statement](#problem-statement)  
3. [Objectives](#objectives)  
4. [Dataset Description](#dataset-description)  
5. [Methodology](#methodology)  
6. [Models and Evaluation](#models-and-evaluation)  
7. [Results](#results)  
8. [Installation](#installation)  
9. [Usage](#usage)  
10. [Future Work](#future-work)  

---

## **1. Introduction**

Forest fires have devastating impacts on ecosystems, economies, and human lives. The **Fire Weather Index (FWI)** is widely used to estimate the risk of forest fires based on meteorological conditions and forest dryness. Accurate FWI predictions are essential for timely interventions and effective resource management.

This project uses machine learning regression models to predict FWI based on weather and FWI component data for two Algerian regions: **Bejaia** and **Sidi Bel-abbes**.

---

## **2. Problem Statement**

Managing forest fires requires accurate and timely prediction of fire risk. Existing systems may lack precision or fail to generalize across regions. This project addresses these gaps by developing a machine learning model capable of accurately predicting FWI using historical data.

---

## **3. Objectives**

1. Develop regression models to predict the Fire Weather Index (FWI).  
2. Identify key features influencing FWI to support decision-making.  
3. Create a scalable and easily deployable system for FWI prediction.  

---

## **4. Dataset Description**

- **Source**: Weather and FWI component data for the Bejaia and Sidi Bel-abbes regions in Algeria (June–September 2012).  
- **Features**:
  - **Weather Variables**: Temperature, Humidity, Wind Speed, Rain.  
  - **FWI Components**: FFMC, DMC, DC, ISI, BUI.  
  - **Target Variable**: Fire Weather Index (FWI).  
- **Size**: 244 observations with 11 variables.

---

## **5. Methodology**

### **1. Data Preprocessing**
- Handled missing values by replacing them with the mean of each column.  
- Scaled numerical features using **StandardScaler** to standardize the dataset.  
- Performed exploratory data analysis (EDA) to understand feature relationships and distributions.  

### **2. Feature Engineering**
- Explored feature importance using models like Random Forest.  
- Created interaction terms and polynomial features to capture non-linear patterns.  

### **3. Model Development**
- Built and evaluated multiple regression models:  
  - Linear Regression  
  - Ridge Regression (with RidgeCV)  
  - Lasso Regression (with LassoCV)  
  - ElasticNet Regression (with ElasticNetCV)  

### **4. Evaluation Metrics**
- **Mean Absolute Error (MAE)**: Measures the average error magnitude.  
- **R² Score**: Measures the proportion of variance explained by the model.  

---

## **6. Models and Evaluation**

| **Model**           | **Mean Absolute Error (MAE)** | **R² Score**       |
|----------------------|-------------------------------|---------------------|
| Linear Regression    | **0.547**                    | **0.985**          |
| Ridge Regression     | 0.564                        | 0.984              |
| RidgeCV              | 0.564                        | 0.984              |
| Lasso Regression     | 1.133                        | 0.949              |
| LassoCV              | 0.620                        | 0.982              |
| ElasticNet           | 1.882                        | 0.875              |
| ElasticNetCV         | 0.658                        | 0.981              |

---

## **7. Results**

1. **Best Model**: Linear Regression performed the best, with an MAE of **0.547** and an R² score of **0.985**.  
2. **Key Features**: FFMC, ISI, temperature, and wind speed were the most influential predictors of FWI.  
3. **Interpretation**: The high R² score indicates that the models captured the data's relationships effectively.  

---

## **8. Installation**

To run the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/algerian-forest-fires-prediction.git
   cd algerian-forest-fires-prediction
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook Algerian_Forest_Fires_Prediction.ipynb
   ```

---

## **9. Usage**

1. **Data Preprocessing**: Automatically cleans and scales the data.  
2. **Model Training**: Train models by running the provided scripts.  
3. **Prediction**: Use the trained model to predict FWI for new inputs.  

### Example:
```python
from sklearn.linear_model import LinearRegression
import pickle

# Load trained model
with open('final_linear_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict new FWI value
new_data = [[85.0, 6.7, 36.3, 12.0, 0.0, 91.0, 112.0, 498.6, 5.1, 22.5]]  # Example input
prediction = model.predict(new_data)
print(f"Predicted FWI: {prediction[0]}")
```

---

## **10. Future Work**

1. Expand the dataset with more recent and diverse data.  
2. Explore advanced machine learning models like XGBoost or neural networks.  
3. Develop a web interface for real-time predictions.  

---

