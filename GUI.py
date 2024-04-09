import tkinter as tk
from tkinter import messagebox

from sklearn.preprocessing import LabelEncoder
import numpy as np

import pandas as pd
import joblib as jl

def y_or_n_to_num(value):
    if value == "yes":
        return 1
    elif value == "male":
        return 1
    else:
        return 0

def preprocess(record):
    age = record[0].get()
    sex = y_or_n_to_num(record[1].get())
    height = float(record[2].get())
    weight = int(record[3].get())
    bmi = weight / height ** 2
    smoker = y_or_n_to_num(record[4].get())
    tobacco = record[5].get()
    alcohol = y_or_n_to_num(record[6].get())
    diabetes = y_or_n_to_num(record[7].get())
    fam_hist = y_or_n_to_num(record[8].get())
    bp_meds = y_or_n_to_num(record[9].get())
    bp = record[10].get()
    heart_rate = record[11].get()
    chol = record[12].get()

    return [age, sex, bmi, smoker, tobacco, alcohol, diabetes, fam_hist, bp_meds, bp, heart_rate, chol]

def get_results(record):
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

    ann_health_cere = jl.load("models/healthy-cerebo/artificial_neural_network.pkl")
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
        "blood_pressure" : [record[9]],
        "cholesterol" : [record[11]],
        "heart_rate" : [record[10]]
    })

    health_art_scaler = jl.load("models/healthy-arterial/ann_scaler.pkl")
    health_art_ann_record = health_art_scaler.transform(health_art_record)

    health_art_le = LabelEncoder()
    health_art_le.fit([0, 3])

    health_cere_record = pd.DataFrame({
        "age" : [record[0]],
        "sex" : [record[1]],
        "smoker" : [record[3]], 
        "tobacco" : [record[4]], 
        "blood_pressure_meds" : [record[8]], 
        "diabetes" : [record[6]], 
        "cholesterol" : [record[11]],
        "blood_pressure" : [record[9]],
        "heart_rate" : [record[10]],
        "bmi" : [record[2]]
    })

    health_cere_scaler = jl.load("models/healthy-cerebo/ann_scaler.pkl")
    health_cere_ann_record = health_cere_scaler.transform(health_cere_record)

    health_cere_le = LabelEncoder()
    health_cere_le.fit([0, 2])

    health_cor_record = pd.DataFrame({
        "age" : [record[0]],
        "smoker" : [record[3]], 
        "tobacco" : [record[4]], 
        "blood_pressure" : [record[9]],
        "family_history" : [record[7]], 
        "bmi" : [record[2]], 
        "alcohol" : [record[5]]
    })

    health_cor_scaler = jl.load("models/healthy-coronary/ann_scaler.pkl")
    health_cor_ann_record = health_cor_scaler.transform(health_cor_record)

    art_cere_cor_record = pd.DataFrame({
        "age" : [record[0]],
        "sex" : [record[1]],
        "blood_pressure" : [record[9]],
        "cholesterol" : [record[11]],
        "heart_rate" : [record[10]]
    })

    art_cere_scaler = jl.load("models/arterial-cerebo/ann_scaler.pkl")
    art_cere_ann_record = art_cere_scaler.transform(art_cere_cor_record)

    art_cere_le = LabelEncoder()
    art_cere_le.fit([2, 3])

    art_cor_scaler = jl.load("models/arterial-coronary/ann_scaler.pkl")
    art_cor_ann_record = art_cor_scaler.transform(art_cere_cor_record)

    art_cor_le = LabelEncoder()
    art_cor_le.fit([1, 3])

    cere_cor_record = pd.DataFrame({
        "age" : [record[0]],
        "smoker" : [record[3]], 
        "tobacco" : [record[4]], 
        "blood_pressure" : [record[9]],
        "bmi" : [record[2]]
    })

    cere_cor_scaler = jl.load("models/cerebo-coronary/ann_scaler.pkl")
    cere_cor_ann_record = cere_cor_scaler.transform(cere_cor_record)

    cere_cor_le = LabelEncoder()
    cere_cor_le.fit([1, 2])

    health_art_results = []
    health_cere_results = []
    health_cor_results = []

    for key, value in health_art_models.items():
        result = int(value.predict(health_art_record))
        health_art_results.append(result)

    result = ann_health_art.predict(health_art_ann_record)
    result = np.where(result > 0.5, 1, 0)
    np.ravel(result)
    result = health_art_le.inverse_transform(result)
    health_art_results.append(result)

    for key, value in health_cere_models.items():
        result = int(value.predict(health_cere_record))
        health_cere_results.append(result)

    result = ann_health_cere.predict(health_cere_ann_record)
    result = np.where(result > 0.5, 1, 0)
    np.ravel(result)
    result = health_cere_le.inverse_transform(result)
    health_cere_results.append(result)

    for key, value in health_cor_models.items():
        result = int(value.predict(health_cor_record))
        health_cor_results.append(result)

    result = ann_health_cor.predict(health_cor_ann_record)
    result = np.where(result > 0.5, 1, 0)
    np.ravel(result)
    #result = health_cor_le.inverse_transform(result)
    health_cor_results.append(result)

    if sum(health_art_results) < 15 and sum(health_cere_results) < 8 and sum(health_cor_results) < 5:
        art_percent = health_art_results.count(3) / 7
        cere_percent = health_cere_results.count(2) / 7
        cor_percent = health_cor_results.count(1) / 7
        healthy_percent = ((1 - art_percent) + (1 - cere_percent) + (1 - cor_percent)) / 21

    elif sum(health_art_results) >= 15 and sum(health_cere_results) < 8 and sum(health_cor_results) < 5:
        art_percent = health_art_results.count(3) / 7
        cere_percent = health_cere_results.count(2) / 7
        cor_percent = health_cor_results.count(1) / 7
        healthy_percent = ((1 - art_percent) + (1 - cere_percent) + (1 - cor_percent)) / 21

    elif sum(health_art_results) < 15 and sum(health_cere_results) >= 8 and sum(health_cor_results) < 5:
        art_percent = health_art_results.count(3) / 7
        cere_percent = health_cere_results.count(2) / 7
        cor_percent = health_cor_results.count(1) / 7
        healthy_percent = ((1 - art_percent) + (1 - cere_percent) + (1 - cor_percent)) / 21

    elif sum(health_art_results) < 15 and sum(health_cere_results) < 8 and sum(health_cor_results) >= 5:
        art_percent = health_art_results.count(3) / 7
        cere_percent = health_cere_results.count(2) / 7
        cor_percent = health_cor_results.count(1) / 7
        healthy_percent = ((1 - art_percent) + (1 - cere_percent) + (1 - cor_percent)) / 21

    elif sum(health_art_results) >= 15 and sum(health_cere_results) >= 8 and sum(health_cor_results) < 5:

        art_cere_results = []

        for key, value in art_cere_models.items():
            result = int(value.predict(art_cere_cor_record))
            art_cere_results.append(result)

        result = ann_art_cere.predict(art_cere_ann_record)
        result = np.where(result > 0.5, 1, 0)
        np.ravel(result)
        result = art_cere_le.inverse_transform(result)
        art_cere_results.append(result)

        if art_cere_results.count(2) >= art_cere_results.count(3):
            print("The classifier has predicted that you may have cerebovasular disease")

        else:
            print("The classifier has predicted that you may have arterial disease")

    elif sum(health_art_results) < 15 and sum(health_cere_results) >= 8 and sum(health_cor_results) >= 5:

        cere_cor_results = []

        for key, value in cere_cor_models.items():
            result = int(value.predict(cere_cor_record))
            cere_cor_results.append(result)

        result = ann_cere_cor.predict(cere_cor_ann_record)
        result = np.where(result > 0.5, 1, 0)
        np.ravel(result)
        result = cere_cor_le.inverse_transform(result)
        art_cere_results.append(result)

        if cere_cor_results.count(2) >= cere_cor_results.count(1):
            print("The classifier has predicted that you may have cerebovasular disease")

        else:
            print("The classifier has predicted that you may have coronary heart disease")

    elif sum(health_art_results) >= 15 and sum(health_cere_results) < 8 and sum(health_cor_results) >= 5:

        art_cor_results = []

        for key, value in art_cor_models.items():
            result = int(value.predict(art_cere_cor_record))
            art_cor_results.append(result)

        result = ann_art_cor.predict(art_cor_ann_record)
        result = np.where(result > 0.5, 1, 0)
        np.ravel(result)
        result = art_cor_le.inverse_transform(result)
        art_cor_results.append(result)

        if art_cor_results.count(3) >= art_cor_results.count(1):
            print("The classifier has predicted that you may have arterial heart disease")

        else:
            print("The classifier has predicted that you may have coronary heart disease")


    else:
        print("The classifier has predicted that you may have coronary heart disease")
        print("The classifier has predicted that you may have cerebovascular disease")
        print("The classifier has predicted that you may have arterial disease")

    return [healthy_percent * 100, cor_percent * 100, cere_percent * 100, art_percent * 100]


def result():
    pre_record = preprocess(entries)
    healthy, coronary, cerebo, arterial = get_results(pre_record)
    healthy_message = f"The classifier predicted a {healthy}% chance of you being healthy"
    coronary_message = f"The classifier predicted a {coronary}% chance of you being at risk of Coronary Heart disease"
    cerebo_message = f"The classifier predicted a {cerebo}% chance of you being at risk of Cerebovascular disease"
    arterial_message = f"The classifier predicted a {arterial}% chance of you being at risk of Cerebovascular disease"
    messagebox.showinfo(title="Results", message=f"{healthy_message}\n{coronary_message}\n{cerebo_message}\n{arterial_message}")

root = tk.Tk()
root.title("Heart Disease Prediction and Classification")

questions = [
    "What is your age?", 
    "What is your sex?", 
    "What is your height? (in metres)", 
    "What is your weight? (in KG)", 
    "Do you smoke? (Y/N)", 
    "How many cigarettes do you smoke per week?", 
    "Do you drink alcohol? (Y/N)", 
    "Do you have diabetes? (Y/N)",  
    "Do you have any relatives that have been diagnosed with Cardiovascular Disease? (Y/N)",
    "Do you take any blood pressure medication? (Y/N)",
    "What is your current systolic blood pressure?",
    "What is your current heart rate?",
    "What is your current cholesterol level?"
    ]
entries = []

for i, question in enumerate(questions):
    tk.Label(root, text=question).grid(row=i, column=0)
    entries.append(tk.Entry(root))
    entries[-1].grid(row=i, column=1)

tk.Button(root, text="Results", command=result).grid(row=len(questions), column=1)


if __name__ == "__main__":
    root.mainloop()