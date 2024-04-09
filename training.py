import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE # SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from collections import Counter

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

# Serialisation

import joblib as jl

# Parsing

import argparse

# Arguments

parser = argparse.ArgumentParser(prog='Model Training')
parser.add_argument('dataset') #Dataset imported into program
parser.add_argument('-f', '--folder') #folder used to find models matched to dataset
args = parser.parse_args()

# Importing data

data = pd.read_csv(args.dataset) # Read in dataset

#Spliting data into features and labels
X = data.drop(columns=['Unnamed: 0', 'label']) #Drop column created by saving dataset
y = data['label']

#Scaling all feature values between 0 and 1
sc = StandardScaler()
X_ann = sc.fit_transform(X)

#Conding label names for artifical neural network
le = LabelEncoder()
le.fit(y)
y_ann = le.transform(y)

#Save scaler for use in testing and gui
jl.dump(value=sc, filename=f"models/{args.folder}/ann_scaler.pkl")

# Train-Test Split and Oversampling

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)
print(Counter(y_train))

sm = SMOTE(random_state=101, k_neighbors=5) #Oversample the datasets
X_smote, y_smote = sm.fit_resample(X_train, y_train)
print(Counter(y_smote))

X_ann_train, X_ann_test, y_ann_train, y_ann_test = train_test_split(X_ann, y_ann, random_state=42, test_size=0.25)
X_ann_smote, y_ann_smote = sm.fit_resample(X_ann_train, y_ann_train)

# Function used to create neural network
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
    model.fit(X_smote, y_smote) #Train models
    print(f"{name} trained.")
    file = f"models/{args.folder}/{name}.pkl"
    jl.dump(value=model, filename=file) #Save model to dataset

model = ann.fit(X_ann_smote, y_ann_smote, epochs=100, verbose=0)
print("artificial_neural_network trained.")
jl.dump(value=ann, filename=f"models/{args.folder}/artificial_neural_network.pkl")