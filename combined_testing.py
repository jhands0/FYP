import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib as jl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Loading each combined dataset
combined_x = pd.read_csv("datasets/combined_data_X.csv")
combined_y = pd.read_csv("datasets/combined_data_y.csv")

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

# Store the results of the combined classifier in a list
pred_list = []

# Count the number of times a majority class was predicted
healthy_count = 0
art_count = 0
cere_count = 0
cor_count = 0

# Iterating over the combined datasets
for index, patient in combined_x.iterrows():

    # Create a subset of the answers to be used in the healthy patient / arterial disease classifier
    health_art_record = pd.DataFrame({
        "age" : [patient['age']],
        "sex" : [patient['sex']],
        "blood_pressure" : [patient['blood_pressure']],
        "cholesterol" : [patient['cholesterol']],
        "heart_rate" : [patient['heart_rate']]
    })

     # Load the standard scaler used to scale the healthy-arterial dataset
    health_art_scaler = jl.load("models/healthy-arterial/ann_scaler.pkl")
    health_art_ann_record = health_art_scaler.transform(health_art_record)

    # Create a label encoder and train it on the class labels
    health_art_le = LabelEncoder()
    health_art_le.fit([0, 3])

    # Create a subset of the answers to be used in the healthy patient / cerebrovascular disease classifier
    health_cere_record = pd.DataFrame({
        "age" : [patient['age']], 
        "sex" : [patient['sex']], 
        "smoker" : [patient['smoker']], 
        "tobacco" : [patient['tobacco']], 
        "blood_pressure_meds" : [patient['blood_pressure_meds']], 
        "diabetes" : [patient['diabetes']], 
        "cholesterol" : [patient['cholesterol']], 
        "blood_pressure" : [patient['blood_pressure']], 
        "heart_rate" : [patient['heart_rate']], 
        "bmi" : [patient['bmi']]
    })

     # Load the standard scaler used to scale the healthy-cerebro dataset
    health_cere_scaler = jl.load("models/healthy-cerebo/ann_scaler.pkl")
    health_cere_ann_record = health_cere_scaler.transform(health_cere_record)

    # Create a label encoder and train it on the class labels
    health_cere_le = LabelEncoder()
    health_cere_le.fit([0, 2])

    # Create a subset of the answers to be used in the healthy patient / coronary heart disease classifier
    health_cor_record = pd.DataFrame({
        "age" : [patient['age']], 
        "smoker" : [patient['smoker']], 
        "tobacco" : [patient['tobacco']], 
        "blood_pressure" : [patient['blood_pressure']], 
        "family_history" : [patient['family_history']], 
        "bmi" : [patient['bmi']], 
        "alcohol" : [patient['alcohol']]
    })

     # Load the standard scaler used to scale the healthy-coronary dataset
    health_cor_scaler = jl.load("models/healthy-coronary/ann_scaler.pkl")
    health_cor_ann_record = health_cor_scaler.transform(health_cor_record)

    # Create a subset of the answers to be used in the arterial patient / cerebrovascular disease and arterial disease / coronary heart disease classifier
    art_cere_cor_record = pd.DataFrame({
        "age" : [patient['age']], 
        "sex" : [patient['sex']], 
        "blood_pressure" : [patient['blood_pressure']], 
        "cholesterol" : [patient['cholesterol']], 
        "heart_rate" : [patient['heart_rate']]
    })

    # Load the standard scaler used to scale the arterial-cerebrovacular dataset
    art_cere_scaler = jl.load("models/arterial-cerebo/ann_scaler.pkl")
    art_cere_ann_record = art_cere_scaler.transform(art_cere_cor_record)

    # Create a label encoder and train it on the class labels
    art_cere_le = LabelEncoder()
    art_cere_le.fit([2, 3])

     # Load the standard scaler used to scale the arterial-coronary dataset
    art_cor_scaler = jl.load("models/arterial-coronary/ann_scaler.pkl")
    art_cor_ann_record = art_cor_scaler.transform(art_cere_cor_record)

    # Create a label encoder and train it on the class labels
    art_cor_le = LabelEncoder()
    art_cor_le.fit([1, 3])

    # Create a subset of the answers to be used in the cerebrovascular disease / coronary heart disease classifier
    cere_cor_record = pd.DataFrame({
        "age" : [patient['age']], 
        "smoker" : [patient['smoker']], 
        "tobacco" : [patient['tobacco']], 
        "blood_pressure" : [patient['blood_pressure']], 
        "bmi" : [patient['bmi']]
    })

     # Load the standard scaler used to scale the cerebro-coronary dataset
    cere_cor_scaler = jl.load("models/cerebo-coronary/ann_scaler.pkl")
    cere_cor_ann_record = cere_cor_scaler.transform(cere_cor_record)

    # Create a lebel encoder and train it on the class labels
    cere_cor_le = LabelEncoder()
    cere_cor_le.fit([1, 2])

    # Store results for each model in the three diagnosis classifiers
    health_art_results = []
    health_cere_results = []
    health_cor_results = []

    # Use each model in the healthy patient / arterial disease classifier to make a prediction
    for key, value in health_art_models.items():
        result = int(value.predict(health_art_record))
        health_art_results.append(result)

    # Make a prediction for the neural network
    result = ann_health_art.predict(health_art_ann_record, verbose=0)
    # Any value greater than 0.5 is 1, anny other value is 0
    result = np.where(result > 0.5, 1, 0)
    np.ravel(result)
    # Conver 0 and 1 to 0 and 3
    result = health_art_le.inverse_transform(result)
    health_art_results.append(result)

    # Use each model in the healthy patient / cerebrovascular disease classifier to make a prediction
    for key, value in health_cere_models.items():
        result = int(value.predict(health_cere_record))
        health_cere_results.append(result)

    # Make a prediction on the neural network
    result = ann_health_cere.predict(health_cere_ann_record, verbose=0)
    # Any value greater than 0.5 is made 1, any other value is made 0
    result = np.where(result > 0.5, 1, 0)
    np.ravel(result)
    # Converts values of 0 and 1 to 0 and 2
    result = health_cere_le.inverse_transform(result)
    health_cere_results.append(result)

    # Use each model in the healthy patient / coronary heart disease classifier to make a prediction
    for key, value in health_cor_models.items():
        result = int(value.predict(health_cor_record))
        health_cor_results.append(result)

    # Makes a prediction for the neural network
    result = ann_health_cor.predict(health_cor_ann_record, verbose=0)
    # Converts any value grater tha 0.5 to 1, any other value to 0
    result = np.where(result > 0.5, 1, 0)
    np.ravel(result)
    health_cor_results.append(result)


    if sum(health_art_results) < 15 and sum(health_cere_results) < 8 and sum(health_cor_results) < 5:
        pred_list.append(0)

        if index < 1289:
            healthy_count += 1

    elif sum(health_art_results) >= 15 and sum(health_cere_results) < 8 and sum(health_cor_results) < 5:
        pred_list.append(3)

        if index >= 2012 and index < 2165:
            art_count += 1

    elif sum(health_art_results) < 15 and sum(health_cere_results) >= 8 and sum(health_cor_results) < 5:
        pred_list.append(2)

        if index >= 1621 and index < 2012:
            cere_count += 1

    elif sum(health_art_results) < 15 and sum(health_cere_results) < 8 and sum(health_cor_results) >= 5:
        pred_list.append(1)

        if index >= 1289 and index < 1621:
            cor_count += 1

    elif sum(health_art_results) >= 15 and sum(health_cere_results) >= 8 and sum(health_cor_results) < 5:

        art_cere_results = []

        # Use each model in the arterial disease / cerebrovacular disease classifier to make a prediction
        for key, value in art_cere_models.items():
            result = int(value.predict(art_cere_cor_record))
            art_cere_results.append(result)

        # Makes a prediction for the neural network
        result = ann_art_cere.predict(art_cere_ann_record, verbose=0)
        # Converts all the values greater than 0.5 to 1, any other values to 0
        result = np.where(result > 0.5, 1, 0)
        np.ravel(result)
        # Converts 0 and 1 to 2 and 3
        result = art_cere_le.inverse_transform(result)
        art_cere_results.append(result)

        if art_cere_results.count(2) >= art_cere_results.count(3):
            pred_list.append(2)

            if index >= 1621 and index < 2012:
                cere_count += 1

        else:
            pred_list.append(3)

            if index >= 2012 and index < 2165:
                art_count += 1

    elif sum(health_art_results) < 15 and sum(health_cere_results) >= 8 and sum(health_cor_results) >= 5:

        cere_cor_results = []

        # Use each model in the cerebrovascular disease / coronary heart disease classifier to make a prediction
        for key, value in cere_cor_models.items():
            result = int(value.predict(cere_cor_record))
            cere_cor_results.append(result)

        # Makes a prediction using the neural network
        result = ann_cere_cor.predict(cere_cor_ann_record, verbose=0)
        # Converts values greater than 0.5 to 1, any other values to 0
        result = np.where(result > 0.5, 1, 0)
        np.ravel(result)
        # Converts 0 and 1 to 1 and 2
        result = cere_cor_le.inverse_transform(result)
        cere_cor_results.append(result)

        if cere_cor_results.count(2) >= cere_cor_results.count(1):
            pred_list.append(2)

            if index >= 1621 and index < 2012:
                cere_count += 1

        else:
            pred_list.append(1)

            if index >= 1289 and index < 1621:
                cor_count += 1

    elif sum(health_art_results) >= 15 and sum(health_cere_results) < 8 and sum(health_cor_results) >= 5:

        art_cor_results = []

        # Use each model in the arterial disease / coronary heart disease classifier to make a prediction
        for key, value in art_cor_models.items():
            result = int(value.predict(art_cere_cor_record))
            art_cor_results.append(result)

        # Makes a prediction using the neural network
        result = ann_art_cor.predict(art_cor_ann_record, verbose=0)
        # Converts all values greater than 0.5 to 1, and other values to 0
        result = np.where(result > 0.5, 1, 0)
        np.ravel(result)
        # Convert 0 and 1 to 1 and 3
        result = art_cor_le.inverse_transform(result)
        art_cor_results.append(result)

        if art_cor_results.count(3) >= art_cor_results.count(1):
            pred_list.append(3)

            if index >= 2012 and index < 2165:
                art_count += 1

        else:
            pred_list.append(1)

            if index >= 1289 and index < 1621:
                cor_count += 1

    else:
        
        pred_list.append(8)

        if index >= 1289 and index < 1621:
            cor_count += 1

        elif index >= 1621 and index < 2012:
            cere_count += 1

        elif index >= 2012 and index < 2165:
            art_count += 1


pred = np.array(pred_list)
real = combined_y.to_numpy()

# Values for all elements in the confusion matrix
health_pred_health = 0
health_pred_cor = 0
health_pred_cere = 0
health_pred_art = 0
cor_pred_health = 0
cor_pred_cor = 0
cor_pred_cere = 0
cor_pred_art = 0
cere_pred_health = 0
cere_pred_cor = 0
cere_pred_cere = 0
cere_pred_art = 0
art_pred_health = 0
art_pred_cor = 0
art_pred_cere = 0
art_pred_art = 0

for i in range(0, len(pred)):
    combined = (pred[i], real[i])
    
    match combined:
    # If the predicted number and actual number are the same, add a true postive
        case (0, 0):
            health_pred_health += 1

        case (1, 1) | (8, 1):
            cor_pred_cor += 1

        case (2, 2) | (8, 2):
            cere_pred_cere += 1
        
        case (3, 3) | (8, 3):
            art_pred_art += 1

    # If the pereicted and acutal don't match, add either a false positive or false negative
        case (0, 1):
            health_pred_cor += 1

        case (0, 2):
            health_pred_cere += 1

        case (0, 3):
            health_pred_art += 1

        case (1, 0):
            cor_pred_health += 1

        case (1, 2):
            cor_pred_cere += 1

        case (1, 3):
            cor_pred_art += 1

        case (2, 0):
            cere_pred_health += 1

        case (2, 1):
            cere_pred_cor += 1

        case (2, 3):
            cere_pred_art += 1

        case (3, 0):
            art_pred_health += 1

        case (3, 1):
            art_pred_cor += 1

        case (3, 2):
            art_pred_cere += 1

# Create a confusion matrix of the results
cm = [[health_pred_health, health_pred_cor, health_pred_cere, health_pred_art],
      [cor_pred_health, cor_pred_cor, cor_pred_cere, cor_pred_art],
      [cere_pred_health, cere_pred_cor, cere_pred_cere, cere_pred_art],
      [art_pred_health, art_pred_cor, art_pred_cere, art_pred_art]]

# Calculate the precision, recall and f1 score for the coronary class
cor_prec = (cor_pred_cor) / ((cor_pred_cor) + (health_pred_cor + cere_pred_cor + art_pred_cor))
cor_rec = (cor_pred_cor) / ((cor_pred_cor) + (cor_pred_health + cor_pred_cere + cor_pred_art))
cor_f1 = (2 * cor_prec * cor_rec) / (cor_prec + cor_rec + 1)

# Calculate the precision, recall and f1 score for the cerebrovasuclar class
cere_prec = (cere_pred_cere) / ((cere_pred_cere) + (health_pred_cere + cor_pred_cere + art_pred_cere))
cere_rec = (cere_pred_cere) / ((cere_pred_cere) + (cere_pred_health + cere_pred_cor + cere_pred_art))
cere_f1 = (2 * cere_prec * cere_rec) / (cere_prec + cere_rec + 1)

# Calculate precision, recall, and f1 score for the arterial class
art_prec = (art_pred_art) / ((art_pred_art) + (health_pred_art + cor_pred_art + cere_pred_art))
art_rec = (art_pred_art) / ((art_pred_art) + (art_pred_health + art_pred_cor + art_pred_cere))
art_f1 = (2 * art_prec * art_rec) / (art_prec + art_rec + 1) 

# Print out precision scores
print("PRECISION")
print(cor_prec * 100)
print(cere_prec * 100)
print(art_prec * 100)

# Print out recall scores
print("\nRECALL")
print(cor_rec * 100)
print(cere_rec * 100)
print(art_rec * 100)

# Print out the f1-scores
print("\nF1 SCORE")
print(cor_f1 * 100)
print(cere_f1 * 100)
print(art_f1 * 100)

# Show the confusion matrix
sns.heatmap(cm, annot=True)
plt.show()