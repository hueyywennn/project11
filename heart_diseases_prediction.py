#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("./heart.csv")
df


# In[3]:


df.isnull().sum()


# In[4]:


df.describe()


# In[5]:


sns.set_style("dark")
sns.set_context("talk")
sns.set_palette("pastel")
sns.despine(left=True, bottom=True)
sns.set(font="monospace")

plt.figure(figsize=(10, 6), dpi=100)
sns.histplot(df['age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# In[6]:


sns.set_style("dark")
sns.set_context("talk")
sns.set_palette("pastel")
sns.despine(left=True, bottom=True)
sns.set(font="monospace")

plt.figure(figsize=(8, 6), dpi=100)
sns.countplot(x=df['sex'], hue=df['target'])
plt.title("Heart Disease Count by Sex")
plt.xlabel("Sex (0 = Female, 1 = Male)")
plt.ylabel("Count")
plt.legend(title="Heart Disease", labels=["No", "Yes"])
plt.show()


# In[7]:


sns.set_style("dark")
sns.set_context("talk")
sns.set_palette("pastel")
sns.despine(left=True, bottom=True)
sns.set(font="monospace")

plt.figure(figsize=(10, 6), dpi=100)
sns.boxplot(x=df['target'], y=df['chol'])
plt.title("Cholesterol Levels by Heart Disease Status")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Cholesterol Level")
plt.show()


# In[8]:


sns.set_style("dark")
sns.set_context("talk")
sns.set_palette("pastel")
sns.despine(left=True, bottom=True)
sns.set(font="monospace")

plt.figure(figsize=(10, 6), dpi=100)
sns.boxplot(x=df['target'], y=df['trestbps'])
plt.title("Resting Blood Pressure by Heart Disease Status")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Resting Blood Pressure")
plt.show()


# In[9]:


sns.set_style("dark")
sns.set_context("talk")
sns.set_palette("pastel")
sns.despine(left=True, bottom=True)
sns.set(font="monospace")

plt.figure(figsize=(10, 6), dpi=100)
sns.boxplot(x=df['target'], y=df['thalach'])
plt.title("Maximum Heart Rate Achieved by Heart Disease Status")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Maximum Heart Rate Achieved")
plt.show()


# In[10]:


sns.set_style("dark")
sns.set_context("talk")
sns.set_palette("pastel")
sns.despine(left=True, bottom=True)
sns.set(font="monospace")

plt.figure(figsize=(15, 10), dpi=100)
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[12]:


X = df.drop(columns=['target'])
y = df['target']


# In[13]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[14]:


# Compare different test sizes and models
test_sizes = [0.1, 0.2, 0.3, 0.4]
results = {}
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "SVM": SVC(kernel='linear', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    results[test_size] = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[test_size][model_name] = accuracy
        print(f"Test Size: {test_size}, Model: {model_name}, Accuracy: {accuracy:.2f}")


# In[15]:


# Find the best test size and model
best_test_size, best_model_name = max(
    [(ts, m) for ts in results for m in results[ts]], key=lambda x: results[x[0]][x[1]]
)
print(f"Best Test Size: {best_test_size}, Best Model: {best_model_name}, Best Accuracy: {results[best_test_size][best_model_name]:.2f}")


# In[16]:


# Final Model with Best Test Size and Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=best_test_size, random_state=42)
final_model = models[best_model_name]
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)


# In[17]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Final Accuracy with {best_model_name} and test size ({best_test_size}): {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[18]:


if hasattr(final_model, 'feature_importances_'):
    feature_importances = pd.Series(final_model.feature_importances_, index=df.columns[:-1]).sort_values(ascending=False)
    plt.figure(figsize=(10, 6), dpi=100)
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title(f"Feature Importance in {best_model_name} Model")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.show()


# In[19]:


plt.figure(figsize=(10, 6), dpi=100)
for model_name in models.keys():
    accuracies = [results[test_size][model_name] for test_size in test_sizes]
    plt.plot(test_sizes, accuracies, marker='o', label=model_name)
plt.title("Model Accuracy Across Different Test Sizes")
plt.xlabel("Test Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

