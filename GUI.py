import tkinter as tk
from tkinter import messagebox

import pandas as pd
import joblib as jl

def y_or_n_to_num(value):
    if value == "y":
        return 1
    elif value == "male":
        return 1
    else:
        return 0

def preprocess(record):
    age = record[0]
    sex = y_or_n_to_num(record[1])
    bmi = record[3] / record[2] ** 2
    chest_pain = y_or_n_to_num(record[4])
    smoker = y_or_n_to_num(record[5])
    tobacco = record[6]
    alcohol = y_or_n_to_num(record[7])
    diabetes = y_or_n_to_num(record[8])
    fam_hist = y_or_n_to_num(record[9])
    bp_meds = y_or_n_to_num(record[10])
    bp = record[11]
    heart_rate = record[12]
    chol = record[13]

    return [age, sex, bmi, chest_pain, smoker, tobacco, alcohol, diabetes, fam_hist, bp_meds, bp, heart_rate, chol]

def get_results(record1, record2):
    ann_health_art = jl.load("models/healthy-arterial/artificial_neural_network.pkl")
    dt_health_art = jl.load("models/healthy-arterial/decision_tree.pkl")
    gnb_health_art = jl.load("models/healthy-arterial/gaussian_naive_bayes.pkl")
    knn_health_art = jl.load("models/healthy-arterial/k_neighbor.pkl")
    lr_health_art = jl.load("models/healthy-arterial/logistic_regression.pkl")
    rf_health_art = jl.load("models/healthy-arterial/random_forest.pkl")
    svm_health_art = jl.load("models/healthy-arterial/support_vector_machine.pkl")

    health_art_models = {
        "Support Vector Machine" : svm_health_art,
        "K-Neighbor" : knn_health_art, 
        "Gaussian Naive Bayes" : gnb_health_art, 
        "Decision Tree" : dt_health_art,
        "Logistic Regression" : lr_health_art,
        "Random Forest" : rf_health_art
    }

    ann_health_cere = jl.load("models/healthy_cerebo/artificial_neural_network.pkl")
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

    ann_health_cor = jl.load("models/healthy-coronary/artificial_neural_network.pkl")
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

    ann_art_cere = jl.load("models/arterial-cerebo/artificial_neural_network.pkl")
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

    ann_cere_cor = jl.load("models/cerebo-coronary/artificial_neural_network.pkl")
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

    ann_art_cor = jl.load("models/arterial-coronary/artificial_neural_network.pkl")
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

    health_art_record = pd.DataFrame({
        "age" : [record[0]],
        "sex" : [record[1]],
        "chest_pain" : [record[3]],
        "blood_pressure" : [record[10]],
        "cholesterol" : [record[12]],
        "heart_rate" : [record[11]]
    })

    health_art_scaler = jl.load("models/healthy-arterial/ann_scaler.pkl")
    health_art_ann_record = health_art_scaler.transform(health_art_record)

    health_cere_record = pd.DataFrame({
        "age" : [record[0]],
        "sex" : [record[1]],
        "smoker" : [record[4]], 
        "tobacco" : [record[5]], 
        "blood_pressure_meds" : [record[9]], 
        "diabetes" : [record[7]], 
        "cholesterol" : [record[12]],
        "blood_pressure" : [record[10]],
        "heart_rate" : [record[11]],
        "bmi" : [record[2]]
    })

    health_cere_scaler = jl.load("models/healthy-cerebo/ann_scaler.pkl")
    health_cere_ann_record = health_cere_scaler.transform(health_cere_record)

    health_cor_record = pd.DataFrame({
        "age" : [record[0]],
        "smoker" : [record[4]], 
        "tobacco" : [record[5]], 
        "blood_pressure" : [record[10]],
        "family_history" : [record[8]], 
        "bmi" : [record[2]], 
        "alcohol" : [record[6]]
    })

    health_cor_scaler = jl.load("models/healthy-coronary/ann_scaler.pkl")
    health_cor_ann_record = health_cor_scaler.transform(health_cor_record)

    art_cere_cor_record = pd.DataFrame({
        "age" : [record[0]],
        "sex" : [record[1]],
        "blood_pressure" : [record[10]],
        "cholesterol" : [record[12]],
        "heart_rate" : [record[11]]
    })

    art_cere_scaler = jl.load("models/arterial-cerebo/ann_scaler.pkl")
    art_cere_ann_record = art_cere_scaler.transform(art_cere_cor_record)

    art_cor_scaler = jl.load("models/arterial-coronary/ann_scaler.pkl")
    art_cor_ann_record = art_cor_scaler.transform(art_cere_cor_record)

    cere_cor_record = pd.DataFrame({
        "age" : [record[0]],
        "smoker" : [record[4]], 
        "tobacco" : [record[5]], 
        "blood_pressure" : [record[10]],
        "bmi" : [record[2]]
    })

    cere_cor_scaler = jl.load("models/cerebo-coronary/ann_scaler.pkl")
    cere_cor_ann_record = cere_cor_scaler.transform(cere_cor_record)

    health_art_results = []
    health_cere_results = []
    health_cor_results = []

    for key, value in health_art_models.items():
        result = int(value.predict(health_art_record))
        health_art_results.append(result)

    for key, value in health_cere_models.items():
        result = int(value.predict(health_cere_record))
        health_cere_results.append(result)

    for key, value in health_cor_models.items():
        result = int(value.predict(health_cor_record))
        health_cor_results.append(result)



def submit():
    pre_record = preprocess(entries)
    ann_pre_record = StandardScaler.fit_transform(pre_record)
    messagebox.showinfo("Results")

root = tk.Tk()
root.title("Heart Disease Prediction and Classification")

labels = [
    "What is your age?", 
    "What is your sex?", 
    "What is your height? (in metres)", 
    "What is your weight? (in KG)", 
    "Do you have any chest pain? (Y/N)", 
    "Do you smoke? (Y/N)", 
    "How many cigarettes do you smoke per week?", 
    "Do you drink alcohol? (Y/N)", 
    "Do you have diabetes? (Y/N)",  
    "Do you have any relatives that have been diagnosed with Cardiovascular Disease? (Y/N)",
    "Do you take any blood pressure medication? (Y/N)",
    "What is your current blood pressure?",
    "What is your current heart rate?",
    "What is your current cholesterol level?"
    ]
entries = []

for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i, column=0)
    entries.append(tk.Entry(root))
    entries[-1].grid(row=i, column=1)

tk.Button(root, text="Results", command=result).grid(row=len(labels), column=1)


if __name__ == "__main__":
    root.mainloop()