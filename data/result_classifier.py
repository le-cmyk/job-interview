import pandas as pd

data_cla = {
    'Accuracy': [0.89, 0.88, 0.89, 0.89, 0.86, 0.88, 0.83, 0.82, 0.82, 0.86, 0.82, 0.81, 0.87, 0.82, 0.77, 0.71, 0.75, 0.65, 0.66, 0.58, 0.45, 0.38, 0.38, 0.41],
    'Balanced Accuracy': [0.85, 0.84, 0.84, 0.83, 0.82, 0.82, 0.81, 0.81, 0.80, 0.80, 0.78, 0.76, 0.76, 0.76, 0.71, 0.69, 0.61, 0.57, 0.57, 0.54, 0.51, 0.39, 0.39, 0.25],
    'ROC AUC': ['None'] * 24,
    'F1 Score': [0.89, 0.88, 0.88, 0.88, 0.86, 0.87, 0.84, 0.83, 0.83, 0.85, 0.82, 0.81, 0.83, 0.83, 0.76, 0.72, 0.69, 0.66, 0.63, 0.58, 0.46, 0.35, 0.35, 0.24],
    'Time Taken': [1.13, 0.72, 2.07, 3.34, 1.81, 4.78, 22.55, 1.31, 2.46, 0.65, 0.53, 1.00, 89.57, 2.15, 0.73, 0.52, 3.85, 2.88, 34.82, 1.24, 0.80, 3.45, 3.64, 0.41]
}

index = [
    'LGBMClassifier', 'DecisionTreeClassifier', 'BaggingClassifier', 'RandomForestClassifier', 'LogisticRegression',
    'ExtraTreesClassifier', 'LinearSVC', 'PassiveAggressiveClassifier', 'LinearDiscriminantAnalysis', 'RidgeClassifier',
    'ExtraTreeClassifier', 'Perceptron', 'CalibratedClassifierCV', 'SGDClassifier', 'BernoulliNB', 'NearestCentroid',
    'AdaBoostClassifier', 'QuadraticDiscriminantAnalysis', 'SVC', 'KNeighborsClassifier', 'GaussianNB',
    'LabelPropagation', 'LabelSpreading', 'DummyClassifier'
]

result_classifier = pd.DataFrame(data_cla, index=index)

