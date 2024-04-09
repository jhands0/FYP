import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import joblib as jl

# Parsing

import argparse

# Arguments

parser = argparse.ArgumentParser(prog='Model Testing')
parser.add_argument('dataset')
parser.add_argument('-f', '--folder')
args = parser.parse_args()

data = pd.read_csv(args.dataset)
classifier = args.folder

X = data.drop(columns=['Unnamed: 0', 'label'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)

# Import models
ann = jl.load(f"models/{classifier}/artificial_neural_network.pkl")

sc = StandardScaler()
X_ann_test = sc.fit_transform(X_test)

le = LabelEncoder()
le.fit(y_test)
y_ann_test = le.transform(y_test)

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
    positive_label = min(y_test)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=positive_label) 
    rec = recall_score(y_test, y_pred, pos_label=positive_label)
    f1 = f1_score(y_test, y_pred, pos_label=positive_label)
    print(f"Accuracy score for {name} is {acc * 100}")
    print(f"Precison score for {name} is {prec * 100}")
    print(f"Recall score for {name} is {rec * 100}")
    print(f"f1 score for {name} is {f1 * 100}")
    #acc, prec, rec, f1 = cross_validate(model, X_test, y_test, scoring=('accuracy', 'precision', 'recall', 'f1'), cv=4)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    cm_plot = sns.heatmap(cm, annot=True)
    cm_plot.figure.savefig(f"models/{classifier}/{name}.png")


results = ann.evaluate(X_ann_test, y_ann_test, batch_size=64, verbose=0)
acc = results[1]
print(f"Accuracy score for artifical_neural_network is {acc * 100}")

predictions = ann.predict(X_ann_test, batch_size=64, verbose=0)
predictions = np.where(predictions > 0.5, 1, 0)
#acc, prec, rec, f1 = cross_validate(ann, X_ann_test, y_ann_test, scoring=('accuracy', 'precision', 'recall', 'f1'), cv=4)
prec = precision_score(y_ann_test, predictions, pos_label=1)
rec = recall_score(y_ann_test, predictions, pos_label=1)
f1 = f1_score(y_ann_test, predictions, pos_label=1)
print(f"Precison score for artificial_neural_network is {prec * 100}")
print(f"Recall score for artifical_neural_network is {rec * 100}")
print(f"f1 score for artificial_neural_network is {f1 * 100}")
cm = confusion_matrix(y_ann_test, predictions)
plt.figure(figsize=(10, 7))
cm_plot = sns.heatmap(cm, annot=True)
cm_plot.figure.savefig(f"models/{classifier}/artificial_neural_network.png")
