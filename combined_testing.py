import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib as jl
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


health_art_record = pd.read_csv("datasets/arterial.csv")
health_art_X = health_art_record.drop(columns=['Unnamed: 0', 'label'])
health_art_y = health_art_record['label']

health_cere_record = pd.read_csv("datasets/cerebo_coronary.csv")
health_cere_X = health_cere_record.drop(columns=['Unnamed: 0', 'label'])
health_cere_y = health_cere_record['label']

health_cor_record = pd.read_csv("datasets/coronary.csv")
health_cor_X = health_cor_record.drop(columns=['Unnamed: 0', 'label'])
health_cor_y = health_cor_record['label']

art_cere_record = pd.read_csv("datasets/classifier_arterial_cerebo.csv")
art_cere_X = art_cere_record.drop(columns=['Unnamed: 0', 'label'])
art_cere_y = art_cere_record['label']

art_cor_record = pd.read_csv("datasets/classifier_arterial_coronary.csv")
art_cor_X = art_cor_record.drop(columns=['Unnamed: 0', 'label'])
art_cor_y = art_cor_record['label']

cere_cor_record = pd.read_csv("datasets/classifier_cerebo_coronary.csv")
cere_cor_X = cere_cor_record.drop(columns=['Unnamed: 0', 'label'])
cere_cor_y = cere_cor_record['label']

# Loading models

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

# Predicting and Scoring

combined_y_real = [*health_art_y, *health_cere_y, *health_cor_y, *art_cere_y, *cere_cor_y, *art_cor_y]

svm_results = []
knn_results = []
gnb_results = []
dt_results = []
lr_results = []
rf_results = []

for key, value in health_art_models.items():
    predicted = value.predict(health_art_X)
    if key == "Support Vector Machine":
        svm_results = [*svm_results, *predicted]
    elif key == "K-Neighbor":
        knn_results = [*knn_results, *predicted]
    elif key == "Gaussian Naive Bayes":
        gnb_results = [*gnb_results, *predicted]
    elif key == "Decision Tree":
        dt_results = [*dt_results, *predicted]
    elif key == "Logistic Regression":
        lr_results = [*lr_results, *predicted]
    elif key == "Random Forest":
        rf_results = [*rf_results, *predicted]

for key, value in health_cere_models.items():
    predicted = value.predict(health_cere_X)
    if key == "Support Vector Machine":
        svm_results = [*svm_results, *predicted]
    elif key == "K-Neighbor":
        knn_results = [*knn_results, *predicted]
    elif key == "Gaussian Naive Bayes":
        gnb_results = [*gnb_results, *predicted]
    elif key == "Decision Tree":
        dt_results = [*dt_results, *predicted]
    elif key == "Logistic Regression":
        lr_results = [*lr_results, *predicted]
    elif key == "Random Forest":
        rf_results = [*rf_results, *predicted]

for key, value in health_cor_models.items():
    predicted = value.predict(health_cor_X)
    if key == "Support Vector Machine":
        svm_results = [*svm_results, *predicted]
    elif key == "K-Neighbor":
        knn_results = [*knn_results, *predicted]
    elif key == "Gaussian Naive Bayes":
        gnb_results = [*gnb_results, *predicted]
    elif key == "Decision Tree":
        dt_results = [*dt_results, *predicted]
    elif key == "Logistic Regression":
        lr_results = [*lr_results, *predicted]
    elif key == "Random Forest":
        rf_results = [*rf_results, *predicted]

for key, value in art_cere_models.items():
    predicted = value.predict(art_cere_X)
    if key == "Support Vector Machine":
        svm_results = [*svm_results, *predicted]
    elif key == "K-Neighbor":
        knn_results = [*knn_results, *predicted]
    elif key == "Gaussian Naive Bayes":
        gnb_results = [*gnb_results, *predicted]
    elif key == "Decision Tree":
        dt_results = [*dt_results, *predicted]
    elif key == "Logistic Regression":
        lr_results = [*lr_results, *predicted]
    elif key == "Random Forest":
        rf_results = [*rf_results, *predicted]

for key, value in cere_cor_models.items():
    predicted = value.predict(cere_cor_X)
    if key == "Support Vector Machine":
        svm_results = [*svm_results, *predicted]
    elif key == "K-Neighbor":
        knn_results = [*knn_results, *predicted]
    elif key == "Gaussian Naive Bayes":
        gnb_results = [*gnb_results, *predicted]
    elif key == "Decision Tree":
        dt_results = [*dt_results, *predicted]
    elif key == "Logistic Regression":
        lr_results = [*lr_results, *predicted]
    elif key == "Random Forest":
        rf_results = [*rf_results, *predicted]

for key, value in art_cor_models.items():
    predicted = value.predict(art_cor_X)
    if key == "Support Vector Machine":
        svm_results = [*svm_results, *predicted]
    elif key == "K-Neighbor":
        knn_results = [*knn_results, *predicted]
    elif key == "Gaussian Naive Bayes":
        gnb_results = [*gnb_results, *predicted]
    elif key == "Decision Tree":
        dt_results = [*dt_results, *predicted]
    elif key == "Logistic Regression":
        lr_results = [*lr_results, *predicted]
    elif key == "Random Forest":
        rf_results = [*rf_results, *predicted]

final_results = {
    "svm" : svm_results,
    "knn" : knn_results, 
    "gnb" : gnb_results, 
    "dt" : dt_results,
    "lr" : lr_results,
    "rf" : rf_results
}

for key, value in final_results.items():
    acc = accuracy_score(combined_y_real, value)
    #pre = precision_score(combined_y_real, value)
    #rec = recall_score(combined_y_real, value)
    cm = confusion_matrix(combined_y_real, value)
    plt.figure(figsize=(10, 7))
    cm_plot = sns.heatmap(cm, annot=True)
    cm_plot.figure.savefig(f"models/{key}_cm.png")
    print(f"Accuracy score for {key} is {acc * 100}")
    #print(f"Precision score for {key} is {pre * 100}")
    #print(f"Recall score for {key} is {rec * 100}")
