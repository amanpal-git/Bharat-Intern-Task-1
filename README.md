# Titanic Survival Prediction Project - TITANIC CLASSIFICATION

## Introduction
This project focuses on predicting whether a passenger survived or not during the tragic sinking of the Titanic. The goal is to build a classification model to predict the survival outcome based on various features such as passenger class, gender, age, siblings/spouses aboard, parents/children aboard, fare, and embarked port.

## Data Loading and Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
df = pd.read_csv("titanic_data.csv")
```

## Exploratory Data Analysis
### Data Overview:

The dataset contains 891 rows and 12 columns.
Important columns: Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.
Data Cleaning:

Handling missing values in columns Age and Embarked.
Removing unnecessary columns: PassengerId, Name, Ticket, Cabin.
Data Visualization:

Visualized survival rate based on gender, class, and age groups.
Explored correlations between features using heatmaps and pair plots.
Feature Engineering
Created a new feature Person categorizing passengers into 'male', 'female', or 'child' categories based on age.
Converted categorical features Sex and Embarked into numeric values.

## Modeling
### Logistic Regression:

Achieved an accuracy of approximately 76.67%.
### Decision Tree Classifier:

Achieved an accuracy of approximately 77.78%.
### K-Nearest Neighbors (KNN) Classifier:

Achieved an accuracy of approximately 68.89%.

## Conclusion
The Decision Tree Classifier model performed the best among the three algorithms, with an accuracy of 77.78%.
Factors such as gender, class, and age played significant roles in predicting survival on the Titanic.
