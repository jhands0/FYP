import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Pre-processing Tools

from sklearn.model_selection import train_test_split, KFold # Splitting dataset
from sklearn.feature_selection import SelectKBest, chi2 # Feature Selection
from imblearn.over_sampling import SMOTE # SMOTE

# Library to fetch dataset

coranary = pd.read_csv("datasets/CHDdata.csv")
cerebo_coranary = pd.read_csv("datasets/CHD_preprocessed.csv")
arterial = pd.read_csv("datasets/HeartDisease.csv")

# EDA
'''
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

'''

# Combining data sets

def normalize(value):
    if value > 0:
        return 1
    elif value == 0:
        return 0
    
def normalize_string(value):
    if value == "Present":
        return 1
    elif value == "Absent":
        return 0
        
def convertKGtoPerWeek(value):
    value = value / 52
    value = value / 0.001
    return value

def convertDaytoWeek(value):
    value = value * 7
    return value

def oneToTwo(value):
    return 2

# Age, Sex, Smoker, Tobacco, BPMeds, Diabetes, Cholestrol, Blood Pressure, Heart Rate, BMI, Label (Glucose maybe?)
cerebo_coranary.loc[cerebo_coranary['prevalentStroke'] == 1, 'TenYearCHD'] = 2
cerebo_coranary.loc[cerebo_coranary['prevalentHyp'] == 1, 'TenYearCHD'] = 2

# Age, Smoker, Tobacco, Blood Pressure, Family History, BMI, Alcohol, Label
final_coranary = pd.DataFrame(columns=['age', 'smoker', 'tobacco', 'blood_pressure', 'family_history', 'bmi', 'alcohol', 'label'])
final_coranary['age'] = coranary['age']
final_coranary['smoker'] = 0
final_coranary['smoker'] = coranary['tobacco'].map(normalize)
final_coranary['tobacco'] = coranary['tobacco'].map(convertKGtoPerWeek)
final_coranary['blood_pressure'] = coranary['sbp']
final_coranary['family_history'] = coranary['famhist'].map(normalize_string)
final_coranary['bmi'] = coranary['obesity']
final_coranary['alcohol'] = coranary['alcohol'].map(normalize)
final_coranary['label'] = coranary['chd']

# Age, Sex, Smoker, Tobacco, BPMeds, Diabetes, Cholesterol, Blood Pressure, Heart Rate, BMI, Label (Glucose maybe?)
final_cerebo_coranary = pd.DataFrame(columns=['age', 'sex', 'smoker', 'tobacco', 'blood_pressure_meds', 'diabetes', 'cholesterol', 'blood_pressure', 'heart_rate', 'bmi', 'label'])
final_cerebo_coranary['age'] = cerebo_coranary['age']
final_cerebo_coranary['sex'] = cerebo_coranary['male']
final_cerebo_coranary['smoker'] = cerebo_coranary['currentSmoker']
final_cerebo_coranary['tobacco'] = cerebo_coranary['cigsPerDay'].map(convertDaytoWeek)
final_cerebo_coranary['blood_pressure_meds'] = cerebo_coranary['BPMeds']
final_cerebo_coranary['diabetes'] = cerebo_coranary['diabetes']
final_cerebo_coranary['cholesterol'] = cerebo_coranary['totChol']
final_cerebo_coranary['blood_pressure'] = cerebo_coranary['sysBP']
final_cerebo_coranary['heart_rate'] = cerebo_coranary['heartRate']
final_cerebo_coranary['bmi'] = cerebo_coranary['BMI']
final_cerebo_coranary['label'] = cerebo_coranary['TenYearCHD']

final_cerebo = final_cerebo_coranary.drop(final_cerebo_coranary[final_cerebo_coranary.label == 1].index)

# Age, Sex, Chest Pain, Blood Pressure, Cholesterol, Blood Sugar, Heart Rate, Label
final_arterial = pd.DataFrame(columns=['age', 'sex', 'chest_pain', 'blood_pressure', 'cholesterol', 'blood_sugar', 'heart_rate', 'label'])
final_arterial['age'] = arterial['age']
final_arterial['sex'] = arterial['sex']
final_arterial['chest_pain'] = arterial['cp'].map(normalize)
final_arterial['blood_pressure'] = arterial['trestbps']
final_arterial['cholesterol'] = arterial['chol']
final_arterial['blood_sugar'] = arterial['fbs']
final_arterial['heart_rate'] = arterial['thalach']
final_arterial['label'] = arterial['num'].map(normalize)

final_arterial.loc[final_arterial['label'] == 1, 'label'] = 3

# Removing NaNs

# Select Features

#new_ds_X = SelectKBest(chi2, k=4).fit_transform(new_ds['age', 'bmi', 'diabetes', 'smoker', 'cigerettes_per_week', 'alcohol'], new_ds['label'])

# Saving preprocessed datasets

final_coranary.to_csv("datasets/coranary.csv")
final_cerebo.to_csv("datasets/cerebo_coranary.csv")
final_arterial.to_csv("datasets/arterial.csv")


visualise_df = pd.DataFrame()
visualise_df['y'] = pd.concat([final_coranary['label'], final_cerebo['label'], final_arterial['label']], ignore_index=True)
visualise_df['x'] = np.zeros(visualise_df['y'].shape)

visualise_df_smote = pd.DataFrame()

sm = SMOTE(random_state=24, k_neighbors=5)
visualise_df_smote['x'], visualise_df_smote['y'] = sm.fit_resample(visualise_df['x'].reshape(-1, 1), visualise_df['y'].reshape(-1, 1))
sns.histplot(visualise_df_smote['y'])
plt.show()

#sns.histplot(final_coranary['label'])
#plt.show()
#sns.histplot(final_cerebo['label'])
#plt.show()
#sns.histplot(final_arterial['label'])
#plt.show()

#sns.heatmap(final_coranary.corr(numeric_only=True))
#plt.show()
#sns.heatmap(final_cerebo.corr(numeric_only=True))
#plt.show()
#sns.heatmap(final_arterial.corr(numeric_only=True))
#plt.show()

# Age, Sex, Blood Pressure, Cholesterol, Heart Rate, Label
classifier_arterial_coranary = pd.DataFrame(columns=['age', 'sex', 'blood_pressure', 'cholesterol', 'heart_rate', 'label'])
true_arterial = final_arterial.drop(final_arterial[final_arterial.label < 1].index)
true_coranary = final_cerebo_coranary.drop(final_cerebo_coranary[final_cerebo_coranary.label == 2].index)
true_coranary = true_coranary.drop(true_coranary[true_coranary.label == 0].index)
true_coranary['label'] = true_coranary['label'].map(oneToTwo)
classifier_arterial_coranary = pd.concat([true_arterial, true_coranary], join="inner")

# Age, Smoker, Tobacco, Blood Pressure, BMI, Label
classifier_cerebo_coranary = pd.DataFrame(columns=['age', 'smoker', 'tobacco', 'blood_pressure', 'bmi', 'label'])
true_coranary = final_coranary.drop(final_coranary[final_coranary.label < 1].index)
true_cerebo_coranary = final_cerebo_coranary.drop(final_cerebo_coranary[final_cerebo_coranary.label < 1].index)
classifier_cerebo_coranary = pd.concat([true_coranary, true_cerebo_coranary], join="inner")

# Age, Sex, Blood Pressure, Cholesterol, Heart Rate, Label
classifier_arterial_cerebo = pd.DataFrame(columns=['age', 'sex', 'blood_pressure', 'cholesterol', 'heart_rate', 'label'])
true_cerebo = final_cerebo_coranary.drop(final_cerebo_coranary[final_cerebo_coranary.label < 2].index)
classifier_arterial_cerebo = pd.concat([true_arterial, true_cerebo], join="inner")

classifier_arterial_coranary.to_csv("datasets/classifier_arterial_coranary.csv")
classifier_cerebo_coranary.to_csv("datasets/classifier_cerebo_coranary.csv")
classifier_arterial_cerebo.to_csv("datasets/classifier_arterial_cerebo.csv")