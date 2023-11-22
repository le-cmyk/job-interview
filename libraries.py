# Streamlit framework
import streamlit as st

# Data Manipulation and Analysis
import pandas as pd  # Pandas: Data manipulation and analysis
import category_encoders as ce  # Category Encoders: Encoders for categorical features

# Data Visualization
import seaborn as sns  # Seaborn: Statistical data visualization
import matplotlib.pyplot as plt  # Matplotlib: Plotting and visualization library

# Machine Learning - Preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder  # Scikit-learn: Preprocessing, feature scaling, and encoding
from sklearn.compose import ColumnTransformer  # Scikit-learn: Column-wise transformations

# Machine Learning - Model Selection and Evaluation
from sklearn.model_selection import train_test_split  # Scikit-learn: Model selection, splitting dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  # Scikit-learn: Model evaluation metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score  # Scikit-learn: Clustering performance metrics

# Machine Learning - Classification Algorithms
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression  # Scikit-learn: Linear models for classification
from sklearn.svm import LinearSVC  # Scikit-learn: Support Vector Machines for classification
from xgboost import XGBClassifier  # XGBoost: Gradient boosting framework for classification
from lightgbm import LGBMClassifier  # LightGBM: Light Gradient Boosting Machine
from lazypredict.Supervised import LazyClassifier  # LazyPredict: Automated model selection and comparison

# Machine Learning - Clustering and Dimensionality Reduction
from sklearn.cluster import KMeans  # Scikit-learn: K-Means clustering algorithm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # Scikit-learn: Linear Discriminant Analysis

# Utilities
from collections import Counter  # Collections: Container datatypes, here for counting objects
import warnings  # Warnings: Handling warnings in Python code
import inspect  #inspect the content of the code
from io import StringIO
from typing import Callable