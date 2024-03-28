import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Pre-processing Tools

from sklearn.model_selection import train_test_split, KFold # Splitting dataset
from sklearn.feature_selection import SelectKBest, chi2 # Feature Selection
from imblearn.over_sampling import SMOTE # SMOTE

# Library to fetch dataset

coronary = pd.read_csv("datasets/CHDdata.csv")
cerebo_coronary = pd.read_csv("datasets/CHD_preprocessed.csv")
arterial = pd.read_csv("datasets/HeartDisease.csv")

# EDA
'''
print(coronary.sample(10))
print(cerebo_coronary.sample(10))
print(arterial.sample(10))

print(coronary.describe())
print(cerebo_coronary.describe())
print(arterial.describe())

print(coronary.isnull().sum())
print(cerebo_coronary.isnull().sum())
print(arterial.isnull().sum())

print(coronary.nunique())
print(cerebo_coronary.nunique())
print(arterial.nunique())

'''
# Removing NaNs

coronary.dropna()
cerebo_coronary.dropna()
arterial.dropna()

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
cerebo_coronary.loc[cerebo_coronary['prevalentStroke'] == 1, 'TenYearCHD'] = 2
cerebo_coronary.loc[cerebo_coronary['prevalentHyp'] == 1, 'TenYearCHD'] = 2

coronary['smoker'] = 0
coronary['smoker'] = coronary['tobacco'].map(normalize)
coronary['tobacco'] = coronary['tobacco'].map(convertKGtoPerWeek)
coronary['famhist'] = coronary['famhist'].map(normalize_string)
coronary['alcohol'] = coronary['alcohol'].map(normalize)

coronary_selector = SelectKBest(chi2, k=7)
coronary_selector.fit(coronary.drop(columns=['chd']), coronary['chd'])
best_coronary = coronary_selector.transform(coronary.drop(columns=['chd']))
print(pd.DataFrame({'columns' : coronary.drop(columns=['chd']).columns, 'kept' : coronary_selector.get_support()}))

'''
# Age, Smoker, Tobacco, Blood Pressure, Family History, BMI, Alcohol, Label
final_coronary = pd.DataFrame(columns=['age', 'smoker', 'tobacco', 'blood_pressure', 'family_history', 'bmi', 'alcohol', 'label'])
final_coronary['age'] = coronary['age']
final_coronary['smoker'] = 0
final_coronary['smoker'] = coronary['tobacco'].map(normalize)
final_coronary['tobacco'] = coronary['tobacco'].map(convertKGtoPerWeek)
final_coronary['blood_pressure'] = coronary['sbp']
final_coronary['family_history'] = coronary['famhist'].map(normalize_string)
final_coronary['bmi'] = coronary['obesity']
final_coronary['alcohol'] = coronary['alcohol'].map(normalize)
final_coronary['label'] = coronary['chd']
'''

cerebo_coronary['cigsPerDay'] = cerebo_coronary['cigsPerDay'].map(convertDaytoWeek)

cerebo_coronary_selector = SelectKBest(chi2, k=10)
cerebo_coronary_selector.fit(cerebo_coronary.drop(columns=['TenYearCHD']), cerebo_coronary['TenYearCHD'])
best_cerebo_coronary = cerebo_coronary_selector.transform(cerebo_coronary.drop(columns=['TenYearCHD']))
print(pd.DataFrame({'columns' : cerebo_coronary.drop(columns=['TenYearCHD']).columns, 'kept' : cerebo_coronary_selector.get_support()}))

'''
# Age, Sex, Smoker, Tobacco, BPMeds, Diabetes, Cholesterol, Blood Pressure, Heart Rate, BMI, Label (Glucose maybe?)
final_cerebo_coronary = pd.DataFrame(columns=['age', 'sex', 'smoker', 'tobacco', 'blood_pressure_meds', 'diabetes', 'cholesterol', 'blood_pressure', 'heart_rate', 'bmi', 'label'])
final_cerebo_coronary['age'] = cerebo_coronary['age']
final_cerebo_coronary['sex'] = cerebo_coronary['male']
final_cerebo_coronary['smoker'] = cerebo_coronary['currentSmoker']
final_cerebo_coronary['tobacco'] = cerebo_coronary['cigsPerDay'].map(convertDaytoWeek)
final_cerebo_coronary['blood_pressure_meds'] = cerebo_coronary['BPMeds']
final_cerebo_coronary['diabetes'] = cerebo_coronary['diabetes']
final_cerebo_coronary['cholesterol'] = cerebo_coronary['totChol']
final_cerebo_coronary['blood_pressure'] = cerebo_coronary['sysBP']
final_cerebo_coronary['heart_rate'] = cerebo_coronary['heartRate']
final_cerebo_coronary['bmi'] = cerebo_coronary['BMI']
final_cerebo_coronary['label'] = cerebo_coronary['TenYearCHD']

final_cerebo = final_cerebo_coronary.drop(final_cerebo_coronary[final_cerebo_coronary.label == 1].index)
'''
arterial['cp'] = arterial['cp'].map(normalize)
arterial['num'] = arterial['num'].map(normalize)

arterial_selector = SelectKBest(chi2, k=7)
arterial_selector.fit(arterial.drop(columns=['num']), arterial['num'])
best_aterial = arterial_selector.transform(arterial.drop(columns=['num']))
print(pd.DataFrame({'columns' : arterial.drop(columns=['num']).columns, 'kept' : arterial_selector.get_support()}))

'''

# Age, Sex, Chest Pain, Blood Pressure, Cholesterol, Blood Sugar, Heart Rate, Label
final_arterial = pd.DataFrame(columns=['age', 'sex', 'chest_pain', 'blood_pressure', 'cholesterol', 'diabetes', 'heart_rate', 'label'])
final_arterial['age'] = arterial['age']
final_arterial['sex'] = arterial['sex']
final_arterial['chest_pain'] = arterial['cp'].map(normalize)
final_arterial['blood_pressure'] = arterial['trestbps']
final_arterial['cholesterol'] = arterial['chol']
final_arterial['diabetes'] = arterial['fbs']
final_arterial['heart_rate'] = arterial['thalach']
final_arterial['label'] = arterial['num'].map(normalize)

final_arterial.loc[final_arterial['label'] == 1, 'label'] = 3

# Saving preprocessed datasets

final_coronary.to_csv("datasets/coronary.csv")
final_cerebo.to_csv("datasets/cerebo_coronary.csv")
final_arterial.to_csv("datasets/arterial.csv")


visualise_df = pd.DataFrame()
visualise_df['y'] = pd.concat([final_coronary['label'], final_cerebo['label'], final_arterial['label']], ignore_index=True)
visualise_df['x'] = np.zeros(visualise_df['y'].shape)

visualise_df_smote = pd.DataFrame()

sm = SMOTE(random_state=24, k_neighbors=5)
visualise_df_smote['x'], visualise_df_smote['y'] = sm.fit_resample(visualise_df['x'].reshape(-1, 1), visualise_df['y'].reshape(-1, 1))
sns.histplot(visualise_df_smote['y'])
plt.show()

#sns.histplot(final_coronary['label'])
#plt.show()
#sns.histplot(final_cerebo['label'])
#plt.show()
#sns.histplot(final_arterial['label'])
#plt.show()

#sns.heatmap(final_coronary.corr(numeric_only=True))
#plt.show()
#sns.heatmap(final_cerebo.corr(numeric_only=True))
#plt.show()
#sns.heatmap(final_arterial.corr(numeric_only=True))
#plt.show()

# Age, Sex, Blood Pressure, Cholesterol, Heart Rate, Label
classifier_arterial_coronary = pd.DataFrame(columns=['age', 'sex', 'blood_pressure', 'cholesterol', 'heart_rate', 'label'])
true_arterial = final_arterial.drop(final_arterial[final_arterial.label < 1].index)
true_coronary = final_cerebo_coronary.drop(final_cerebo_coronary[final_cerebo_coronary.label == 2].index)
true_coronary = true_coronary.drop(true_coronary[true_coronary.label == 0].index)
true_coronary['label'] = true_coronary['label'].map(oneToTwo)
classifier_arterial_coronary = pd.concat([true_arterial, true_coronary], join="inner")

# Age, Smoker, Tobacco, Blood Pressure, BMI, Label
classifier_cerebo_coronary = pd.DataFrame(columns=['age', 'smoker', 'tobacco', 'blood_pressure', 'bmi', 'label'])
true_coronary = final_coronary.drop(final_coronary[final_coronary.label < 1].index)
true_cerebo_coronary = final_cerebo_coronary.drop(final_cerebo_coronary[final_cerebo_coronary.label < 1].index)
classifier_cerebo_coronary = pd.concat([true_coronary, true_cerebo_coronary], join="inner")

# Age, Sex, Blood Pressure, Cholesterol, Heart Rate, Label
classifier_arterial_cerebo = pd.DataFrame(columns=['age', 'sex', 'blood_pressure', 'cholesterol', 'diabetes' 'heart_rate', 'label'])
true_cerebo = final_cerebo_coronary.drop(final_cerebo_coronary[final_cerebo_coronary.label < 2].index)
classifier_arterial_cerebo = pd.concat([true_arterial, true_cerebo], join="inner")

classifier_arterial_coronary.to_csv("datasets/classifier_arterial_coronary.csv")
classifier_cerebo_coronary.to_csv("datasets/classifier_cerebo_coronary.csv")
classifier_arterial_cerebo.to_csv("datasets/classifier_arterial_cerebo.csv")

'''