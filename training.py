import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from imblearn.over_sampling import SMOTE # SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Potential Models

import keras #ANN
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.svm import SVC # Support Vector Machine
from sklearn.neighbors import KNeighborsClassifier # K Neighbour Classifier
from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayes
from sklearn.tree import DecisionTreeClassifier # Decision Tree
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.ensemble import RandomForestClassifier # Random Forest

# Serialisation

import joblib as jl

# Parsing

import argparse

# Arguments

parser = argparse.ArgumentParser(prog='Model Training')
parser.add_argument('dataset')
parser.add_argument('-f', '--folder')
args = parser.parse_args()

# Importing data

data = pd.read_csv(args.dataset)

X = data.drop(columns=['Unnamed: 0', 'label'])
y = data['label']

sc = StandardScaler()
X_ann = sc.fit_transform(X)

le = LabelEncoder()
le.fit(y)
y_ann = le.transform(y)

# Cross Validation and Oversampling

kf = KFold(n_splits=5, shuffle=True, random_state=101)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)
print(Counter(y_train))

sm = SMOTE(random_state=101, k_neighbors=5)
X_smote, y_smote = sm.fit_resample(X_train, y_train)
print(Counter(y_smote))

X_ann_train, X_ann_test, y_ann_train, y_ann_test = train_test_split(X_ann, y_ann, random_state=42, test_size=0.25)
X_ann_smote, y_ann_smote = sm.fit_resample(X_ann_train, y_ann_train)

def create_ANN():
    ann = Sequential()
    ann.add(Dense(16, input_dim=len(X.columns), activation='relu'))
    ann.add(Dense(12, activation='relu'))
    ann.add(Dense(1, activation='sigmoid'))
    ann.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return ann

ann = create_ANN()

svm = SVC()

knn = KNeighborsClassifier()

gnb = GaussianNB()

dt = DecisionTreeClassifier()

lr = LogisticRegression()

rf = RandomForestClassifier()

models = {
    "support_vector_machine" : svm,
    "k_neighbor" : knn, 
    "gaussian_naive_bayes" : gnb, 
    "decision_tree" : dt,
    "logistic_regression" : lr,
    "random_forest" : rf
}

# Model Training and Serialization

for name, model in models.items():
    model.fit(X_smote, y_smote)
    results = cross_val_score(model, X_smote, y_smote, cv=kf)
    print(f"{name} trained.")
    #print(f"{name} accuracy: {results}")
    #print(f"{name} mean accuracy: {results.mean()}")
    file = f"models/{args.folder}/{name}.pkl"
    jl.dump(value=model, filename=file)

model = ann.fit(X_ann_smote, y_ann_smote, epochs=100, verbose=0)
print("artificial_neural_network trained.")
#print(f"artificial_neural_network accuracy: {results}")
#print(f"artificial_neural_network mean accuracy: {results.mean()}")
jl.dump(value=ann, filename=f"models/{args.folder}/artificial_neural_network.pkl")