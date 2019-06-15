# Titanic
Kaggle competition to analyse and predict what sort of traveller would survive the sinking of the Titanic.
Here, I implemented Stacking (a specific ensembling of base models) of multiple machine learning algorithms (RandomForest, SVM, Logistic Regression) to create my model.

I started off by first importing my libraries: 
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


