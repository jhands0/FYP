import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Deserialization

import joblib as jl

# GUI application

import tkinter as tk


def y_or_n_to_num(value):
    '''
    Function used to binaraize string values for yes, no, male and female
    '''
    if value == "y":
        return 1
    elif value == "male":
        return 1
    else:
        return 0

# Questionare questions with input error handling

age = int(input("\nWhat is your age? "))

sex = str(input("\nWhat is your sex? "))
sex = y_or_n_to_num(sex)

height = float(input("\nWhat is your height? (in metres) "))

weight = float(input("\nWhat is your weight? (in kg) "))

bmi = weight / height ** 2

smoker = str(input("\nDo you smoke? (y/n) "))

if smoker == "y":
    smoker_num = 1
    cigs_per_week = int(input("\nHow many cigerettes do you have per week? "))
    # Skip the question regarding cigerettes if the user said no to smoking question
else:
    smoker_num = 0
    cigs_per_week = 0

alcohol = str(input("\nDo you drink alcohol? (y/n) "))
alcohol = y_or_n_to_num(alcohol)

diabetes = str(input("\nDo you have diabetes? (y/n) "))
diabetes = y_or_n_to_num(diabetes)

fam_hist = str(input("\nDo you have any relatives that have been diagnosed with cardiovasular disease? (y/n) "))
fam_hist = y_or_n_to_num(fam_hist)

bp_meds = str(input("\nDo you take any blood pressure medication? (y/n) "))
bp_meds = y_or_n_to_num(bp_meds)

bp = int(input("\nWhat is your current systolic blood pressure? "))

heart_rate = int(input("\nWhat is your active heart rate? "))

chol = int(input("\nWhat is your current cholesterol value? "))
print("\n")


# Record all answers in a dataframe
record = pd.DataFrame({
    "age" : [age], 
    "sex" : [sex], 
    "smoker" : [smoker_num], 
    "tobacco" : [cigs_per_week], 
    "blood_pressure" : [bp], 
    "blood_pressure_meds" : [bp_meds], 
    "diabetes" : [diabetes], 
    "cholesterol" : [chol], 
    "heart_rate" : [heart_rate], 
    "family_history" : [fam_hist], 
    "bmi" : [bmi], 
    "alcohol" : [alcohol]
})

# Create a subset of the answers to be used in the healthy patient / arterial disease classifier
health_art_record = pd.DataFrame({
    "age" : [age],
    "sex" : [sex],
    "blood_pressure" : [bp],
    "cholesterol" : [chol],
    "heart_rate" : [heart_rate]
})

# Create a subset of the answers to be used in the healthy patient / cerebovascular disease classifier
health_cere_record = pd.DataFrame({
    "age" : [age], 
    "sex" : [sex], 
    "smoker" : [smoker_num], 
    "tobacco" : [cigs_per_week], 
    "blood_pressure_meds" : [bp_meds], 
    "diabetes" : [diabetes], 
    "cholesterol" : [chol], 
    "blood_pressure" : [bp], 
    "heart_rate" : [heart_rate], 
    "bmi" : [bmi]
})

# Create a subset of the answers to be used in the healthy patient / coronary heart disease classifier
health_cor_record = pd.DataFrame({
    "age" : [age], 
    "smoker" : [smoker_num], 
    "tobacco" : [cigs_per_week], 
    "blood_pressure" : [bp], 
    "family_history" : [fam_hist], 
    "bmi" : [bmi], 
    "alcohol" : [alcohol]
})

# Create a subset of the answers to be used in the arterial disease / cerebovascular disease classifier
art_cere_record = pd.DataFrame({
    "age" : [age], 
    "sex" : [sex], 
    "blood_pressure" : [bp], 
    "cholesterol" : [chol], 
    "heart_rate" : [heart_rate]
})

# Create a subset of the answers to be used in the arterial disease / coronary heart disease classifier
art_cor_record = pd.DataFrame({
    "age" : [age], 
    "sex" : [sex], 
    "blood_pressure" : [bp], 
    "cholesterol" : [chol], 
    "heart_rate" : [heart_rate]
})

# Create a subset of the answers to be used in the cerebovasuclar disease / coronary heart disease classifier
cere_cor_record = pd.DataFrame({
    "age" : [age], 
    "smoker" : [smoker_num], 
    "tobacco" : [cigs_per_week], 
    "blood_pressure" : [bp], 
    "bmi" : [bmi]
})

# Loading models

dt_health_art = jl.load("models/healthy-arterial/decision_tree.pkl")
gnb_health_art = jl.load("models/healthy-arterial/gaussian_naive_bayes.pkl")
knn_health_art = jl.load("models/healthy-arterial/k_neighbor.pkl")
lr_health_art = jl.load("models/healthy-arterial/logistic_regression.pkl")
rf_health_art = jl.load("models/healthy-arterial/random_forest.pkl")
svm_health_art = jl.load("models/healthy-arterial/support_vector_machine.pkl")

#Create a separate dictionary for every classifier with names
health_art_models = {
    "Support Vector Machine" : svm_health_art,
    "K-Neighbor" : knn_health_art, 
    "Gaussian Naive Bayes" : gnb_health_art, 
    "Decision Tree" : dt_health_art,
    "Logistic Regression" : lr_health_art,
    "Random Forest" : rf_health_art
}

dt_health_cere = jl.load("models/healthy-cerebo/decision_tree.pkl")
gnb_health_cere = jl.load("models/healthy-cerebo/gaussian_naive_bayes.pkl")
knn_health_cere = jl.load("models/healthy-cerebo/k_neighbor.pkl")
lr_health_cere = jl.load("models/healthy-cerebo/logistic_regression.pkl")
rf_health_cere = jl.load("models/healthy-cerebo/random_forest.pkl")
svm_health_cere = jl.load("models/healthy-cerebo/support_vector_machine.pkl")

health_cere_models = {
    "Support Vector Machine" : svm_health_cere,
    "K-Neighbor" : knn_health_cere, 
    "Gaussian Naive Bayes" : gnb_health_cere, 
    "Decision Tree" : dt_health_cere,
    "Logistic Regression" : lr_health_cere,
    "Random Forest" : rf_health_cere
}

dt_health_cor = jl.load("models/healthy-coronary/decision_tree.pkl")
gnb_health_cor = jl.load("models/healthy-coronary/gaussian_naive_bayes.pkl")
knn_health_cor = jl.load("models/healthy-coronary/k_neighbor.pkl")
lr_health_cor = jl.load("models/healthy-coronary/logistic_regression.pkl")
rf_health_cor = jl.load("models/healthy-coronary/random_forest.pkl")
svm_health_cor = jl.load("models/healthy-coronary/support_vector_machine.pkl")

health_cor_models = {
    "Support Vector Machine" : svm_health_cor,
    "K-Neighbor" : knn_health_cor, 
    "Gaussian Naive Bayes" : gnb_health_cor, 
    "Decision Tree" : dt_health_cor,
    "Logistic Regression" : lr_health_cor,
    "Random Forest" : rf_health_cor
}

# Predicting

# Records used to hold the results of the models in the diagnois phase classifiers
health_art_results = []
health_cere_results = []
health_cor_results = []

# Enumerate over healthy patient / arterial disease and make a prediction for each model, adding it to the results list
for key, value in health_art_models.items():
    result = int(value.predict(health_art_record))
    health_art_results.append(result)

# Enumerate over healthy patient / cerebovasular disease and make a prediction for each model, adding it to the results list
for key, value in health_cere_models.items():
    result = int(value.predict(health_cere_record))
    health_cere_results.append(result)

# Enumerate over healthy patient / coronary heart disease and make a prediction for each model, adding it to the results list
for key, value in health_cor_models.items():
    result = int(value.predict(health_cor_record))
    health_cor_results.append(result)

print(health_art_results)
print(health_cere_results)
print(health_cor_results)


if sum(health_art_results) < 12 and sum(health_cere_results) < 8 and sum(health_cor_results) < 4:
    print("You are predicted healthy")

elif sum(health_art_results) >= 12 and sum(health_cere_results) < 8 and sum(health_cor_results) < 4:
    print("The classifier has predicted that you may have arterial disease")

elif sum(health_art_results) < 12 and sum(health_cere_results) >= 8 and sum(health_cor_results) < 4:
    print("The classifier has predicted that you may have cerebovasular disease")

elif sum(health_art_results) < 12 and sum(health_cere_results) < 8 and sum(health_cor_results) >= 4:
    print("The classifier has predicted that you may have coronary heart disease")

elif sum(health_art_results) >= 12 and sum(health_cere_results) >= 8 and sum(health_cor_results) < 4:

    # Load in the additional classifier needed for arterial disease / cerebovasular disease classification
    dt_art_cere = jl.load("models/arterial-cerebo/decision_tree.pkl")
    gnb_art_cere = jl.load("models/arterial-cerebo/gaussian_naive_bayes.pkl")
    knn_art_cere = jl.load("models/arterial-cerebo/k_neighbor.pkl")
    lr_art_cere = jl.load("models/arterial-cerebo/logistic_regression.pkl")
    rf_art_cere = jl.load("models/arterial-cerebo/random_forest.pkl")
    svm_art_cere = jl.load("models/arterial-cerebo/support_vector_machine.pkl")

    art_cere_models = {
        "Support Vector Machine" : svm_art_cere,
        "K-Neighbor" : knn_art_cere, 
        "Gaussian Naive Bayes" : gnb_art_cere, 
        "Decision Tree" : dt_art_cere,
        "Logistic Regression" : lr_art_cere,
        "Random Forest" : rf_art_cere
    }

    art_cere_results = []

    # Enumerates over the arterial disease / cerebovascular disease classifiers and makes a prediction for each model, adding it to the results list
    for key, value in art_cere_models.items():
        result = int(value.predict(art_cere_record))
        art_cere_results.append(result)

    print("\n")
    print(art_cere_results)

elif sum(health_art_results) < 12 and sum(health_cere_results) >= 8 and sum(health_cor_results) >= 4:

    # Load in the additional classifier needed for coronary heart disease / cerebovasular disease classification
    dt_cere_cor = jl.load("models/cerebo-coronary/decision_tree.pkl")
    gnb_cere_cor = jl.load("models/cerebo-coronary/gaussian_naive_bayes.pkl")
    knn_cere_cor = jl.load("models/cerebo-coronary/k_neighbor.pkl")
    lr_cere_cor = jl.load("models/cerebo-coronary/logistic_regression.pkl")
    rf_cere_cor = jl.load("models/cerebo-coronary/random_forest.pkl")
    svm_cere_cor = jl.load("models/cerebo-coronary/support_vector_machine.pkl")

    cere_cor_models = {
        "Support Vector Machine" : svm_cere_cor,
        "K-Neighbor" : knn_cere_cor, 
        "Gaussian Naive Bayes" : gnb_cere_cor, 
        "Decision Tree" : dt_cere_cor,
        "Logistic Regression" : lr_cere_cor,
        "Random Forest" : rf_cere_cor
    }

    cere_cor_results = []

    # Enumerates over coronary heart disease / cerebovascular disease classifier and makes a prediction for each model, adding it to the results list
    for key, value in cere_cor_models.items():
        result = int(value.predict(cere_cor_record))
        cere_cor_results.append(result)

    print("\n")
    print(cere_cor_results)

elif sum(health_art_results) >= 12 and sum(health_cere_results) < 8 and sum(health_cor_results) >= 4:

    # Load in the additional classifier needed for coronary heart disease / arterial disease classification
    dt_art_cor = jl.load("models/arterial-coronary/decision_tree.pkl")
    gnb_art_cor = jl.load("models/arterial-coronary/gaussian_naive_bayes.pkl")
    knn_art_cor = jl.load("models/arterial-coronary/k_neighbor.pkl")
    lr_art_cor = jl.load("models/arterial-coronary/logistic_regression.pkl")
    rf_art_cor = jl.load("models/arterial-coronary/random_forest.pkl")
    svm_art_cor = jl.load("models/arterial-coronary/support_vector_machine.pkl")

    art_cor_models = {
        "Support Vector Machine" : svm_art_cor,
        "K-Neighbor" : knn_art_cor, 
        "Gaussian Naive Bayes" : gnb_art_cor, 
        "Decision Tree" : dt_art_cor,
        "Logistic Regression" : lr_art_cor,
        "Random Forest" : rf_art_cor
    }

    art_cor_results = []

    # Enumerates over coronary heart disease / arterial disease classifier and makes a prediction for each model, adding it to the results list
    for key, value in art_cor_models.items():
        result = int(value.predict(art_cor_record))
        art_cor_results.append(result)

    print("\n")
    print(art_cor_results)