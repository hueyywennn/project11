# Heart Diseases 
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Project Overview
develop a predictive model for heart disease diagnosis using machine learning techniques.

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

### Features
1. **Data Cleaning**
  -	Checked missing values: none
2. **Exploratory Data Analysis (EDA)**
  -	Statistical Summaries: mean, median, variance, and standard deviation
  -	Correlation Analysis: heatmap correlations
  -	Distribution Plots: histograms, bar plots,
  -	Outlier Detection: boxplot
3. **Machine Learning Models**
  -	Data Normalization: Standard Scaler
  -	Classification Algorithms: Random Forest, Logistic Regression, Support Vector Machine (SVM), Gradient Boosting
  -	Predictive Modeling: Classification
  -	Test sizes: 10%, 20%, 30%, 40%
4. **Interactive Visualizations**
  -	Classification Visualization: Model accuracy across different test sizes

## Tools used
1. **Programming Language** 
  - Python
2. **Libraries**
  - pandas, numpy, scikit-learn, matplotlib
3. **Visualization Tools**
  - plotly, seaborn
