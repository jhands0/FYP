import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE # SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Potential Models

import keras #ANN
from keras.models import Sequential
from keras.layers import Dense

from sklearn.svm import SVC # Support Vector Machine
from sklearn.neighbors import KNeighborsClassifier # K Neighbour Classifier
from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayes
from sklearn.tree import DecisionTreeClassifier # Decision Tree
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.ensemble import RandomForestClassifier # Random Forest

from skorch import NeuralNetClassifier

# Serialisation

import joblib as jl

# Importing data

data = pd.read_csv("datasets/classifier_cerebo_coronary.csv")
classifier = "cerebo-coronary"

X = data.drop(columns=['Unnamed: 0', 'label'])
y = data['label']

sc = StandardScaler()
X_ann = sc.fit_transform(X)

le = LabelEncoder()
le.fit(y)
y_ann = le.transform(y)

# Train Test Split and Oversampling

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)
sm = SMOTE(random_state=24, k_neighbors=5)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

X_ann_train, X_ann_test, y_ann_train, y_ann_test = train_test_split(X_ann, y_ann, random_state=42, test_size=0.25)
X_ann_train_smote, y_ann_train_smote = sm.fit_resample(X_ann_train, y_ann_train)

ann = Sequential()
ann.add(Dense(16, input_dim=len(X.columns), activation='relu'))
ann.add(Dense(12, activation='relu'))
ann.add(Dense(1, activation='sigmoid'))
ann.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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
    model.fit(X_train_smote, y_train_smote)
    print(f"{name} trained.")
    file = f"models/{classifier}/{name}.pkl"
    jl.dump(value=model, filename=file)

model = ann.fit(X_ann_train_smote, y_ann_train_smote, epochs=100, batch_size=64)
print("artificial_neural_network trained.")
jl.dump(value=ann, filename=f"models/{classifier}/artificial_neural_network.pkl")