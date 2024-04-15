import tkinter as tk
from tkinter import messagebox

from sklearn.preprocessing import LabelEncoder
import numpy as np

import pandas as pd
import joblib as jl

def y_or_n_to_num(value):
    '''
    Function used to binaraize string values for yes, no and male
    '''
    if value == "yes" or value == "y" or value == "Y" or value == "Yes":
        return 1
    elif value == "male" or value == "m" or value == "Male" or value == "M":
        return 1
    else:
        return 0

def preprocess(record):
    '''
    Function to convert all user input into a format that the classifier can read
    '''

    # Parsing in all user input from the tkinter entry boxes, doing type checking for all values
    age = int(record[0].get())
    sex = y_or_n_to_num(str(record[1].get()))
    height = float(record[2].get())
    weight = float(record[3].get())
    bmi = weight / height ** 2 # Calculate the BMI from height and weight
    smoker = y_or_n_to_num(str(record[4].get()))
    tobacco = float(record[5].get())
    alcohol = y_or_n_to_num(str(record[6].get()))
    diabetes = y_or_n_to_num(str(record[7].get()))
    fam_hist = y_or_n_to_num(str(record[8].get()))
    bp_meds = y_or_n_to_num(str(record[9].get()))
    bp = float(record[10].get())
    heart_rate = float(record[11].get())
    chol = float(record[12].get())

    # Return an array containing all the features in the order they were asked
    return [age, sex, bmi, smoker, tobacco, alcohol, diabetes, fam_hist, bp_meds, bp, heart_rate, chol]

def get_results(record):
    '''
    This large function loads in every classifier and model used to make predictions on the user data

    It also converts the user record to six different dataframes for the six classifiers, removing uneeded user input when necessary

    Uses the dataframes as input to the classifiers

    Uses an if-else statement to aggregate the scores of the classifiers by seeing if they pass number of model thresholds

    If the diagnosis classifier thinks that the person has a type of CVD, test record on a classifier in the distingusing phase

    Returns four scores for each class, Healthy, CHD, Cerebrovascular, and Arterial
    '''

    # Loading models for each classifier
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

    # Create a subset of the answers to be used in the healthy patient / arterial disease classifier
    health_art_record = pd.DataFrame({
        "age" : [record[0]],
        "sex" : [record[1]],
        "blood_pressure" : [record[9]],
        "cholesterol" : [record[11]],
        "heart_rate" : [record[10]]
    })

    # Load the standard scaler used to scale the healthy-arterial dataset
    health_art_scaler = jl.load("models/healthy-arterial/ann_scaler.pkl")
    health_art_ann_record = health_art_scaler.transform(health_art_record)

    # Load in label encoder and train it
    health_art_le = LabelEncoder()
    health_art_le.fit([0, 3])

    # Create a subset of the answers to be used in the healthy patient / cerebovascular disease classifier
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

    # Load the standard scaler used to scale the healthy-cerebo dataset
    health_cere_scaler = jl.load("models/healthy-cerebo/ann_scaler.pkl")
    health_cere_ann_record = health_cere_scaler.transform(health_cere_record)

    # Load in label encoder and train it
    health_cere_le = LabelEncoder()
    health_cere_le.fit([0, 2])

    # Create a subset of the answers to be used in the healthy patient / coronary heart disease classifier
    health_cor_record = pd.DataFrame({
        "age" : [record[0]],
        "smoker" : [record[3]], 
        "tobacco" : [record[4]], 
        "blood_pressure" : [record[9]],
        "family_history" : [record[7]], 
        "bmi" : [record[2]], 
        "alcohol" : [record[5]]
    })

    # Load the standard scaler used to scale the healthy-coronary dataset
    health_cor_scaler = jl.load("models/healthy-coronary/ann_scaler.pkl")
    health_cor_ann_record = health_cor_scaler.transform(health_cor_record)

    # Create a subset of the answers to be used in the arterial disease / cerebovascular disease and arterial disease / cerebovascular disease classifier
    art_cere_cor_record = pd.DataFrame({
        "age" : [record[0]],
        "sex" : [record[1]],
        "blood_pressure" : [record[9]],
        "cholesterol" : [record[11]],
        "heart_rate" : [record[10]]
    })

    # Load the standard scaler used to scale the arterial-cerebo dataset
    art_cere_scaler = jl.load("models/arterial-cerebo/ann_scaler.pkl")
    art_cere_ann_record = art_cere_scaler.transform(art_cere_cor_record)

    # Load in label encoder and train it
    art_cere_le = LabelEncoder()
    art_cere_le.fit([2, 3])

    # Load the standard scaler used to scale the arterial-coronary dataset
    art_cor_scaler = jl.load("models/arterial-coronary/ann_scaler.pkl")
    art_cor_ann_record = art_cor_scaler.transform(art_cere_cor_record)

    # Load in label encoder and train it
    art_cor_le = LabelEncoder()
    art_cor_le.fit([1, 3])

    # Create a subset of the answers to be used in the cerebovasuclar disease / coronary heart disease classifier
    cere_cor_record = pd.DataFrame({
        "age" : [record[0]],
        "smoker" : [record[3]], 
        "tobacco" : [record[4]], 
        "blood_pressure" : [record[9]],
        "bmi" : [record[2]]
    })

    # Load the standard scaler used to scale the cerebo-coronary dataset
    cere_cor_scaler = jl.load("models/cerebo-coronary/ann_scaler.pkl")
    cere_cor_ann_record = cere_cor_scaler.transform(cere_cor_record)

    # Load in label encoder and train it
    cere_cor_le = LabelEncoder()
    cere_cor_le.fit([1, 2])

    # Create lists to store results of each diagnosis classifier
    health_art_results = []
    health_cere_results = []
    health_cor_results = []

    # Enumerate over models and make predictions for healthy arterial classifier
    for key, value in health_art_models.items():
        result = int(value.predict(health_art_record))
        health_art_results.append(result)

    # Make a prediction for the neural network
    result = ann_health_art.predict(health_art_ann_record, verbose=0)
    # Convert any values greater than 0.5 to 1, anything else to 0
    result = np.where(result > 0.5, 1, 0)
    np.ravel(result)
    # Convert 0 and 1 to 0 and 3
    result = health_art_le.inverse_transform(result)
    health_art_results.append(result)

    # Enumerate over models and make predictions for healthy cerebovascular classifier
    for key, value in health_cere_models.items():
        result = int(value.predict(health_cere_record))
        health_cere_results.append(result)

    # Make a prediction for the neural network
    result = ann_health_cere.predict(health_cere_ann_record, verbose=0)
    # Convert any values greater than 0.5 to 1, anything else to 0
    result = np.where(result > 0.5, 1, 0)
    np.ravel(result)
    # Convert 0 and 1 to 0 and 2
    result = health_cere_le.inverse_transform(result)
    health_cere_results.append(result)

    # Enumerate over models and make predictions for healthy coronary classifier
    for key, value in health_cor_models.items():
        result = int(value.predict(health_cor_record))
        health_cor_results.append(result)

    # Make a prediction for the neural network
    result = ann_health_cor.predict(health_cor_ann_record, verbose=0)
    # Convert any values greater than 0.5 to 1, anything else to 0
    result = np.where(result > 0.5, 1, 0)
    np.ravel(result)
    health_cor_results.append(result)

    # Compare the diagnosis classifier results to classifier thresholds
    if sum(health_art_results) < 15 and sum(health_cere_results) < 8 and sum(health_cor_results) < 5:
        # Calculate the scores by dividing the number of occurances of the class in the classifiers by the number of models the class can appear in 
        art_score = health_art_results.count(3) / 7
        cere_score = health_cere_results.count(2) / 7
        cor_score = health_cor_results.count(1) / 7
        healthy_score = (health_art_results.count(0) + health_cere_results.count(0) + health_cor_results.count(0)) / 21

    # Compare the diagnosis classifier results to classifier thresholds
    elif sum(health_art_results) >= 15 and sum(health_cere_results) < 8 and sum(health_cor_results) < 5:
        # Calculate the scores by dividing the number of occurances of the class in the classifiers by the number of models the class can appear in 
        art_score = health_art_results.count(3) / 7
        cere_score = health_cere_results.count(2) / 7
        cor_score = health_cor_results.count(1) / 7
        healthy_score = (health_art_results.count(0) + health_cere_results.count(0) + health_cor_results.count(0)) / 21

    # Compare the diagnosis classifier results to classifier thresholds
    elif sum(health_art_results) < 15 and sum(health_cere_results) >= 8 and sum(health_cor_results) < 5:
        # Calculate the scores by dividing the number of occurances of the class in the classifiers by the number of models the class can appear in 
        art_score = health_art_results.count(3) / 7
        cere_score = health_cere_results.count(2) / 7
        cor_score = health_cor_results.count(1) / 7
        healthy_score = (health_art_results.count(0) + health_cere_results.count(0) + health_cor_results.count(0)) / 21

    # Compare the diagnosis classifier results to classifier thresholds
    elif sum(health_art_results) < 15 and sum(health_cere_results) < 8 and sum(health_cor_results) >= 5:
        # Calculate the scores by dividing the number of occurances of the class in the classifiers by the number of models the class can appear in 
        art_score = health_art_results.count(3) / 7
        cere_score = health_cere_results.count(2) / 7
        cor_score = health_cor_results.count(1) / 7
        healthy_score = (health_art_results.count(0) + health_cere_results.count(0) + health_cor_results.count(0)) / 21

    # Compare the diagnosis classifier results to classifier thresholds
    elif sum(health_art_results) >= 15 and sum(health_cere_results) >= 8 and sum(health_cor_results) < 5:

        art_cere_results = []

        # Enumerate over models and make predictions for arterial disease cerebovascular disease classifier
        for key, value in art_cere_models.items():
            result = int(value.predict(art_cere_cor_record))
            art_cere_results.append(result)

        # Make a prediction for the neural network
        result = ann_art_cere.predict(art_cere_ann_record, verbose=0)
        # Convert any values greater than 0.5 to 1, anything else to 0
        result = np.where(result > 0.5, 1, 0)
        np.ravel(result)
        # Convert 0 and 1 to 2 and 3
        result = art_cere_le.inverse_transform(result)
        art_cere_results.append(result)

        # Calculate the scores by dividing the number of occurances of the class in the classifiers by the number of models the class can appear in 
        art_score = (health_art_results.count(3) + art_cere_results.count(3)) / 14
        cere_score = (health_cere_results.count(2) + art_cere_results.count(2)) / 14
        cor_score = health_cor_results.count(1) / 7
        healthy_score = (health_art_results.count(0) + health_cere_results.count(0) + health_cor_results.count(0)) / 21

    # Compare the diagnosis classifier results to classifier thresholds
    elif sum(health_art_results) < 15 and sum(health_cere_results) >= 8 and sum(health_cor_results) >= 5:

        cere_cor_results = []

        # Enumerate over models and make predictions for arterial disease cerebovascular disease classifier
        for key, value in cere_cor_models.items():
            result = int(value.predict(cere_cor_record))
            cere_cor_results.append(result)

        # Make a prediction for the neural network
        result = ann_cere_cor.predict(cere_cor_ann_record, verbose=0)
        # Convert any values greater than 0.5 to 1, anything else to 0
        result = np.where(result > 0.5, 1, 0)
        np.ravel(result)
        #Convert 0 and 1 to 1 and 2
        result = cere_cor_le.inverse_transform(result)
        cere_cor_results.append(result)

        # Calculate the scores by dividing the number of occurances of the class in the classifiers by the number of models the class can appear in 
        art_score = health_art_results.count(3) / 7
        cere_score = (health_cere_results.count(2) + cere_cor_results.count(2)) / 14
        cor_score = (health_cor_results.count(1) + cere_cor_results.count(1)) / 14
        healthy_score = (health_art_results.count(0) + health_cere_results.count(0) + health_cor_results.count(0)) / 21

    # Compare the diagnosis classifier results to classifier thresholds
    elif sum(health_art_results) >= 15 and sum(health_cere_results) < 8 and sum(health_cor_results) >= 5:

        art_cor_results = []

        # Enumerate over models and make predictions for arterial disease cerebovascular disease classifier
        for key, value in art_cor_models.items():
            result = int(value.predict(art_cere_cor_record))
            art_cor_results.append(result)

        # Make a prediction for the neural network
        result = ann_art_cor.predict(art_cor_ann_record, verbose=0)
        # Convert any values greater than 0.5 to 1, anything else to 0
        result = np.where(result > 0.5, 1, 0)
        np.ravel(result)
        #Convert 0 and 1 to 1 and 3
        result = art_cor_le.inverse_transform(result)
        art_cor_results.append(result)

        # Calculate the scores by dividing the number of occurances of the class in the classifiers by the number of models the class can appear in 
        art_score = (health_art_results.count(3) + art_cor_results.count(3)) / 14
        cere_score = health_cere_results.count(2) / 7
        cor_score = (health_cor_results.count(1) + art_cor_results.count(1)) / 14
        healthy_score = (health_art_results.count(0) + health_cere_results.count(0) + health_cor_results.count(0)) / 21


    else:
        # Calculate the scores by dividing the number of occurances of the class in the classifiers by the number of models the class can appear in 
        art_score = health_art_results.count(3) / 7
        cere_score = health_cere_results.count(2) / 7
        cor_score = health_cor_results.count(1) / 7
        healthy_score = (health_art_results.count(0) + health_cere_results.count(0) + health_cor_results.count(0)) / 21

    return [healthy_score * 100, cor_score * 100, cere_score * 100, art_score * 100]


def result():
    '''
    Function ran when the Results button is pressed on the GUI, preprocesses the input data, runs the combined classifier,
    preprocesses the result, and sets an output message using the messagebox tkinter widget
    '''
    # Preprocesses the input record using type checking in the preprocess function
    pre_record = preprocess(entries)

    # Runs the combined classifier 
    healthy_percent, coronary_percent, cerebo_percent, arterial_percent = get_results(pre_record)

    # Converts all the percentage scores to 2 decimal places
    healthy_percent = '{0:.2f}'.format(healthy_percent) 
    coronary_percent = '{0:.2f}'.format(coronary_percent)
    cerebo_percent = '{0:.2f}'.format(cerebo_percent)
    arterial_percent = '{0:.2f}'.format(arterial_percent)

    # Creates human readable messages of the percentages for an end user  
    healthy_message = f"The classifier predicted a {healthy_percent}% chance of you being healthy."
    coronary_message = f"The classifier predicted a {coronary_percent}% chance of you being at risk of Coronary Heart disease."
    cerebo_message = f"The classifier predicted a {cerebo_percent}% chance of you being at risk of Cerebovascular disease."
    arterial_message = f"The classifier predicted a {arterial_percent}% chance of you being at risk of Arterial disease."
    messagebox.showinfo(title="Results", message=f"{healthy_message}\n\n{coronary_message}\n\n{cerebo_message}\n\n{arterial_message}")

# Creates GUI and adds a fitting title
root = tk.Tk()
root.title("Heart Disease Prediction and Classification")

# List of questions that will be added to the GUI in a grid alongside entry boxes
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

for i, question in enumerate(questions): #For every question asked
    # Make a tkinter label containing the question in row i, column 0
    tk.Label(root, text=question).grid(row=i, column=0)

    # Add a tkinter entry object to the entries list
    entries.append(tk.Entry(root))
    entries[-1].grid(row=i, column=1)

tk.Button(root, text="Results", command=result).grid(row=13, column=1)

# Starts the GUI when the script is ran
if __name__ == "__main__":
    root.mainloop()