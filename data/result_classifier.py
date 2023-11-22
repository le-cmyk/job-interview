import pandas as pd

# Créer un dictionnaire avec les données
data_cla = {
    'Accuracy': [0.90, 0.89, 0.88, 0.86, 0.88, 0.89, 0.86, 0.88, 0.86, 0.88, 0.88, 0.85, 0.87, 0.82, 0.82, 0.80, 0.78, 0.73, 0.75, 0.69, 0.65, 0.58, 0.45, 0.38, 0.38, 0.41],
    'Balanced Accuracy': [0.86, 0.85, 0.84, 0.83, 0.82, 0.82, 0.82, 0.82, 0.82, 0.81, 0.81, 0.79, 0.78, 0.77, 0.76, 0.76, 0.71, 0.71, 0.62, 0.59, 0.56, 0.54, 0.50, 0.40, 0.40, 0.25],
    'F1 Score': [0.90, 0.89, 0.88, 0.86, 0.88, 0.88, 0.86, 0.87, 0.86, 0.87, 0.87, 0.84, 0.85, 0.82, 0.82, 0.80, 0.77, 0.73, 0.70, 0.68, 0.63, 0.58, 0.46, 0.36, 0.36, 0.24],
    'Time Taken': [15.24, 1.03, 0.74, 2.16, 2.23, 3.18, 19.10, 4.38, 1.22, 0.60, 1.78, 1.44, 75.40, 0.44, 2.51, 0.82, 0.73, 0.45, 3.70, 2.75, 32.71, 1.04, 0.69, 3.18, 2.90, 0.39]
}

index=[
    'XGBClassifier', 'LGBMClassifier', 'DecisionTreeClassifier',
    'LinearDiscriminantAnalysis', 'BaggingClassifier',
    'RandomForestClassifier', 'LinearSVC', 'ExtraTreesClassifier',
    'LogisticRegression', 'RidgeClassifier',
    'RidgeClassifierCV', 'PassiveAggressiveClassifier',
    'CalibratedClassifierCV', 'ExtraTreeClassifier',
    'SGDClassifier', 'Perceptron', 'BernoulliNB',
    'NearestCentroid', 'AdaBoostClassifier',
    'QuadraticDiscriminantAnalysis', 'SVC', 'KNeighborsClassifier',
    'GaussianNB', 'LabelSpreading', 'LabelPropagation',
    'DummyClassifier'
]
# Créer le DataFrame "result_classifier"
result_classifier = pd.DataFrame(data_cla, index=index)

#result of the grid search on lgbm
best_params = {'learning_rate': 0.1, 'max_depth': 3, 'min_child_samples': 10, 'n_estimators': 200, 'subsample': 0.8}

best_score = 0.8999134199134199

