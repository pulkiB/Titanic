# Titanic
Kaggle competition to analyse and predict what sort of traveller would survive the sinking of the Titanic.
Here, I implemented Stacking (a specific ensembling of base models) of multiple machine learning algorithms (RandomForest, SVM, Logistic Regression) to create my model.

## importing my libraries: 
```
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.cross_validation import KFold
```

## Feature exploration and cleaning:
 1. Sorted out the features I believed to be relevant 
 2. Engineered additional features to aid analysis
 3. Got rid of unnecessary data

## Data Visualization
  1. Used seaborn to create a heatmap of the Pearson Correlation of features. This helped understand which features were most relevant contributors to survival.
  2. Created pairplots to visualise the distribution of data between features
  
## Model Creation
  1. Built a logistic regression model.
  2. To account for overfitting, I approached the model building using stacking. Here we built decision trees, random forest, svm and logistic regression for the first level predictions.
  
 Credit for this structure must be given to Anisotropic, who provided a detailed introduction to ensembling in Python on Kaggle -- https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python/notebook 
