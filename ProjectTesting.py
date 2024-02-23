import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Pre-processing Tools

from sklearn.model_selection import train_test_split, KFold # Splitting dataset
from imblearn.over_sampling import SMOTE # SMOTE
from sklearn.feature_selection import SelectKBest, chi2 # Feature Selection

# Potential Models

from keras import layers # Multi-Level Perceptron / Artifical Neural Network
from sklearn.neighbors import KNeighborsClassifier # K Neighbour Classifier
from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayes
from sklearn import svm # Support Vector Machine
from sklearn.tree import DecisionTreeClassifier # Decision Tree
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.ensemble import RandomForestClassifier # Random Forest

# Wrapper to allow MLP/ANN to interact with sklearn modules

from scikeras.wrappers import KerasClassifier

coranary = pd.read_csv("datasets/CHDdata.csv")
cerebo_coranary = pd.read_csv("datasets/CHD_preprocessed.csv")
arterial = pd.read_csv("datasets/arterial_disease_and_IBD_EHRs_from_France.csv")


# EDA

print(coranary.sample(10))
print(cerebo_coranary.sample(10))
print(arterial.sample(10))

print(coranary.describe())
print(cerebo_coranary.describe())
print(arterial.describe())

print(coranary.isnull().sum())
print(cerebo_coranary.isnull().sum())
print(arterial.isnull().sum())

print(coranary.nunique())
print(cerebo_coranary.nunique())
print(arterial.nunique())

sns.histplot(coranary['chd'])#plt.show()
sns.histplot(cerebo_coranary['TenYearCHD'])

sns.heatmap(coranary.corr(numeric_only=True))
plt.show()
#sns.heatmap(cerebo_coranary.corr())
#sns.heatmap(arterial.corr())

# Combining data sets

'''
def changeSmoker(value):
    if value >= 1:
        return 1
    elif value < 1:
        return 0
        
def convertKGtoPerWeek(value):
    value = value / 52
    value = value / 13.2
    return value

coranary['smoker'] = 0
coranary['smoker'] = map(changeSmoker, coranary['tobacco'])
coranary['tobacco'] = map(convertKGtoPerWeek, coranary['tobacco'])
cerebo_coranary['TenYearCHD'].loc[cerebo_coranary['prevalentStroke'] == 1] = 2
cerebo_coranary['TenYearCHD'].loc[cerebo_coranary['prevalentHyp'] == 1] = 2
cerebo_coranary['alchohol'] = 0

new_ds = pd.DataFrame(columns=['age', 'smoker', 'cigerettes_per_week', 'alchohol', 'BMI', 'label'])
new_ds['age'] = coranary['age']
new_ds['smoker'] = coranary['tobacco']
new_ds['cigerettes_per_week'] = coranary['tobacco']
new_ds['alchohol'] = coranary['alchohol']
new_ds['BMI'] = coranary['obesity']
new_ds['label'] = coranary['chd']

print(new_ds.head())
#new_ds['age'] = new_ds['age'] + 
'''

# Select Features

# Removing NaNs

# Train Test Split

'''
ann = layers.Sequential([
        layers.Dense(6, activation="relu", name="Input Layer"),
        layers.Dense(12, activation="relu", name="Hidden Layer"),
        layers.Dense(4, activation='softmax', name="Output/Class Layer"),
    ])
ann.compile(loss='')


knn = KNeighborsClassifier()

gnb = GaussianNB()

dt = DecisionTreeClassifier()

lr = LogisticRegression()

rf = RandomForestClassifier()

models = [ann, knn, gnb, dt, lr, rf]
'''