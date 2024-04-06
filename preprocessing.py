import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Pre-processing Tools

from sklearn.model_selection import train_test_split, KFold # Splitting dataset
from sklearn.feature_selection import SelectKBest, chi2 # Feature Selection
from imblearn.over_sampling import SMOTE # SMOTE

coronary = pd.read_csv("datasets/CHDdata.csv")
cerebo_coronary = pd.read_csv("datasets/CHD_preprocessed.csv")
arterial = pd.read_csv("datasets/HeartDisease.csv")
arterial = arterial.drop(columns=['Unnamed: 0'])

#sns.histplot(coronary, x="chd")
#plt.show()
#sns.histplot(cerebo_coronary, x="TenYearCHD")
#plt.show()
#sns.histplot(arterial, x="num")
#plt.show()

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

coronary.dropna(inplace=True)
cerebo_coronary.dropna(inplace=True)
arterial.dropna(inplace=True)

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

def max_to_active(value):
    return value * 0.77

# Age, Sex, Smoker, Tobacco, BPMeds, Diabetes, Cholestrol, Blood Pressure, Heart Rate, BMI, Label (Glucose maybe?)
cerebo_coronary.loc[((cerebo_coronary.prevalentStroke == 1) & (cerebo_coronary.TenYearCHD == 1)), 'TenYearCHD'] = 2
cerebo_coronary.loc[((cerebo_coronary.prevalentHyp == 1) & (cerebo_coronary.TenYearCHD == 1)), 'TenYearCHD'] = 2

coronary['smoker'] = 0
coronary['smoker'] = coronary['tobacco'].map(normalize)
coronary['tobacco'] = coronary['tobacco'].map(convertKGtoPerWeek)
coronary['famhist'] = coronary['famhist'].map(normalize_string)
coronary['alcohol'] = coronary['alcohol'].map(normalize)

#x = coronary.drop(columns=['chd'])
#coronary_scores, _ = chi2(x, coronary['chd'])
#sns.barplot(x = x.columns, y = coronary_scores)
#plt.show()

# Age, Smoker, Tobacco, Blood Pressure, Family History, BMI, Alcohol, Label
final_coronary = pd.DataFrame(columns=['age', 'smoker', 'tobacco', 'blood_pressure', 'family_history', 'bmi', 'alcohol', 'label'])
final_coronary['age'] = coronary['age']
final_coronary['smoker'] = coronary['smoker']
final_coronary['tobacco'] = coronary['tobacco']
final_coronary['blood_pressure'] = coronary['sbp']
final_coronary['family_history'] = coronary['famhist']
final_coronary['bmi'] = coronary['obesity']
final_coronary['alcohol'] = coronary['alcohol']
final_coronary['label'] = coronary['chd']

final_coronary_corr = final_coronary.corr()
sns.heatmap(final_coronary_corr)
plt.show()

cerebo_coronary['cigsPerDay'] = cerebo_coronary['cigsPerDay'].map(convertDaytoWeek)

#x = cerebo_coronary.drop(columns=['TenYearCHD'])
#cerebo_coronary_scores, _ = chi2(x, cerebo_coronary['TenYearCHD'])
#sns.barplot(x = x.columns, y = cerebo_coronary_scores)
#plt.show()

# Age, Sex, Smoker, Tobacco, BPMeds, Diabetes, Cholesterol, Blood Pressure, Heart Rate, BMI, Label (Glucose maybe?)
final_cerebo_coronary = pd.DataFrame(columns=['age', 'sex', 'smoker', 'tobacco', 'blood_pressure_meds', 'diabetes', 'cholesterol', 'blood_pressure', 'heart_rate', 'bmi', 'label'])
final_cerebo_coronary['age'] = cerebo_coronary['age']
final_cerebo_coronary['sex'] = cerebo_coronary['male']
final_cerebo_coronary['smoker'] = cerebo_coronary['currentSmoker']
final_cerebo_coronary['tobacco'] = cerebo_coronary['cigsPerDay']
final_cerebo_coronary['blood_pressure_meds'] = cerebo_coronary['BPMeds']
final_cerebo_coronary['diabetes'] = cerebo_coronary['diabetes']
final_cerebo_coronary['cholesterol'] = cerebo_coronary['totChol']
final_cerebo_coronary['blood_pressure'] = cerebo_coronary['sysBP']
final_cerebo_coronary['heart_rate'] = cerebo_coronary['heartRate']
final_cerebo_coronary['bmi'] = cerebo_coronary['BMI']
final_cerebo_coronary['label'] = cerebo_coronary['TenYearCHD']

final_cerebo = final_cerebo_coronary.drop(final_cerebo_coronary[final_cerebo_coronary.label == 1].index)

final_cerebo_corr = final_cerebo.corr()
sns.heatmap(final_cerebo_corr)
plt.show()

arterial['cp'] = arterial['cp'].map(normalize)
arterial['thalach'] = arterial['thalach'].map(max_to_active)
arterial['num'] = arterial['num'].map(normalize)

#x = arterial.drop(columns=['num'])
#arterial_scores, _ = chi2(x, arterial['num'])
#sns.barplot(x = x.columns, y = arterial_scores)
#plt.show()


# Age, Sex, Chest Pain, Blood Pressure, Cholesterol, Blood Sugar, Heart Rate, Label
final_arterial = pd.DataFrame(columns=['age', 'sex', 'chest_pain', 'blood_pressure', 'cholesterol', 'heart_rate', 'label'])
final_arterial['age'] = arterial['age']
final_arterial['sex'] = arterial['sex']
final_arterial['chest_pain'] = arterial['cp']
final_arterial['blood_pressure'] = arterial['trestbps']
final_arterial['cholesterol'] = arterial['chol']
final_arterial['heart_rate'] = arterial['thalach']
final_arterial['label'] = arterial['num']

final_arterial.loc[final_arterial['label'] == 1, 'label'] = 3

final_arterial_corr = final_arterial.corr()
sns.heatmap(final_arterial_corr)
plt.show()

# Saving preprocessed datasets

final_coronary.to_csv("datasets/coronary.csv")
final_cerebo.to_csv("datasets/cerebo_coronary.csv")
final_arterial.to_csv("datasets/arterial.csv")

sns.histplot(final_coronary, x="label")
plt.show()
sns.histplot(final_cerebo, x="label")
plt.show()
sns.histplot(final_arterial, x="label")
plt.show()

'''
visualise_df = pd.DataFrame()
visualise_df['y'] = pd.concat([final_coronary['label'], final_cerebo['label'], final_arterial['label']], ignore_index=True)
visualise_df['x'] = np.zeros(visualise_df['y'].shape)

visualise_df_smote = pd.DataFrame()

sm = SMOTE(random_state=24, k_neighbors=5)
visualise_df_smote['x'], visualise_df_smote['y'] = sm.fit_resample(visualise_df['x'].reshape(-1, 1), visualise_df['y'].reshape(-1, 1))
sns.histplot(visualise_df_smote['y'])
plt.show()
'''

# Age, Sex, Blood Pressure, Cholesterol, Heart Rate, Label
classifier_arterial_coronary = pd.DataFrame(columns=['age', 'sex', 'blood_pressure', 'heart_rate', 'label'])
true_arterial = final_arterial.drop(final_arterial[final_arterial.label < 1].index)
true_coronary = final_cerebo_coronary.drop(final_cerebo_coronary[final_cerebo_coronary.label == 2].index)
true_coronary = true_coronary.drop(true_coronary[true_coronary.label == 0].index)
classifier_arterial_coronary = pd.concat([true_arterial, true_coronary], join="inner")

# Age, Smoker, Tobacco, Blood Pressure, BMI, Label
classifier_cerebo_coronary = pd.DataFrame(columns=['age', 'tobacco', 'blood_pressure', 'bmi', 'label'])
true_coronary = final_coronary.drop(final_coronary[final_coronary.label < 1].index)
true_cerebo_coronary = final_cerebo_coronary.drop(final_cerebo_coronary[final_cerebo_coronary.label < 1].index)
classifier_cerebo_coronary = pd.concat([true_coronary, true_cerebo_coronary], join="inner")

# Age, Sex, Blood Pressure, Cholesterol, Heart Rate, Label
classifier_arterial_cerebo = pd.DataFrame(columns=['age', 'sex', 'blood_pressure', 'cholesterol', 'heart_rate', 'label'])
true_cerebo = final_cerebo_coronary.drop(final_cerebo_coronary[final_cerebo_coronary.label < 2].index)
classifier_arterial_cerebo = pd.concat([true_arterial, true_cerebo], join="inner")

classifier_arterial_coronary.to_csv("datasets/classifier_arterial_coronary.csv")
classifier_cerebo_coronary.to_csv("datasets/classifier_cerebo_coronary.csv")
classifier_arterial_cerebo.to_csv("datasets/classifier_arterial_cerebo.csv")