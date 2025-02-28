# Heart Diseases Identification
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Project Overview
This project aims to develop a predictive model for heart disease diagnosis using machine learning techniques. By analyzing key medical attributes, the model can help detect the presence of heart disease and assist healthcare professionals in early diagnosis and treatment.

### Dataset 
(https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data)
| Name            | Description                                                                                              |
|----------------|----------------------------------------------------------------------------------------------------------|
| age            | Age of the patient                                                                                       |
| sex            | Sex of the patient (0 = female, 1 = male)                                                                |
| cp             | Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)       |
| trestbps       | Resting blood pressure (in mm Hg)                                                                        |
| chol          | Serum cholesterol in mg/dl                                                                              |
| fbs           | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)                                                   |
| restecg       | Resting electrocardiographic results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy) |
| thalach       | Maximum heart rate achieved                                                                             |
| exang         | Exercise-induced angina (1 = yes, 0 = no)                                                               |
| oldpeak       | ST depression induced by exercise relative to rest                                                      |
| slope         | The slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)                    |
| ca            | Number of major vessels (0-3) colored by fluoroscopy                                                    |
| thal          | Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)                                       |
| target        | Diagnosis of heart disease (1 = presence, 0 = absence)                                                  |

## Project Objectives
1. **Data Cleaning & Preprocessing**: Handling missing values, feature normalization, and categorical encoding.
2. **Exploratory Data Analysis (EDA)**: Understanding the relationships between different medical attributes and heart disease.
3. **Feature Engineering**: Selecting and transforming features to improve model performance.
4. **Model Training & Evaluation**: Implementing and comparing multiple machine learning models for heart disease prediction.
5. **Hyperparameter Tuning**: Optimizing model parameters with trial test sizes of 10%, 20%, 30%, and 40%.

## Machine Learning Models Used
- **Random Forest Classifier**: Ensemble method for improved accuracy.
- **Logistic Regression**: Baseline model for classification.
- **Support Vector Machine (SVM)**: Effective for high-dimensional spaces.
- **Gradient Boosting**: Boosting technique for performance optimization.
- **Standard Scaler**: Used for feature normalization.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy for data manipulation
- **Visualization Tools**: Matplotlib, Seaborn for data exploration

## Project Workflow
1. **Data Ingestion**: Load and preprocess the dataset.
2. **Exploratory Data Analysis (EDA)**: Analyze and visualize medical feature relationships.
3. **Feature Engineering**: Normalize numerical data.
4. **Model Training**: Train Random Forest, Logistic Regression, SVM, and Gradient Boosting models.
5. **Hyperparameter Tuning**: Optimize models using different test sizes (10%, 20%, 30%, 40%).
6. **Model Evaluation**: Compare model performance using evaluation metrics.
7. **Results Interpretation**: Derive insights and recommendations from predictions.
