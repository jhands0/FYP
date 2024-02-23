import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import joblib as jl

dataset = "coranary"
classifier = "healthy-coranary"

data = pd.read_csv(f"datasets/{dataset}.csv")

X = data.drop(columns=['Unnamed: 0', 'label'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)

# Import models

dt = jl.load(f"models/{classifier}/decision_tree.pkl")
gnb = jl.load(f"models/{classifier}/gaussian_naive_bayes.pkl")
knn = jl.load(f"models/{classifier}/k_neighbor.pkl")
lr = jl.load(f"models/{classifier}/logistic_regression.pkl")
rf = jl.load(f"models/{classifier}/random_forest.pkl")
svm = jl.load(f"models/{classifier}/support_vector_machine.pkl")

models = {"decision_tree" : dt, 
          "gaussian_naive_bayes" : gnb, 
          "k_neighbor" : knn, 
          "logistic_regression" : lr, 
          "random_forest" : rf, 
          "support_vector_machine" : svm}

# Predictions

for name, model in models.items():
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"Accuracy score for {name} is {score * 100}")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    cm_plot = sns.heatmap(cm, annot=True)
    cm_plot.figure.savefig(f"models/{classifier}/{name}.png")