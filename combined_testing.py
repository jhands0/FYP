import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib as jl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random

'''
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
'''

combined_x = pd.read_csv("datasets/fake_data_X.csv")
combined_y = pd.read_csv("datasets/fake_data_y.csv")

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

def formatResultsArt(value):
    if value == 1:
        return "Arterial Disease"
    elif value == 2:
        return "Cerebovascular Disease"
    
def formatResultsCor(value):
    if value == 1:
        return "Coronary Heart Disease"
    elif value == 2:
        return "Cerebovasular Disease"

pred_list = []
count = ''
healthy_count = 0
art_count = 0
cere_count = 0
cor_count = 0

for index, patient in combined_x.iterrows():

    '''
    record = pd.DataFrame({
        "age" : [patient.age], 
        "sex" : [patient.sex], 
        "smoker" : [patient.smoker], 
        "tobacco" : [patient.tobacco], 
        "blood_pressure" : [patient.blood_pressure], 
        "blood_pressure_meds" : [patient.blood_pressure_meds], 
        "diabetes" : [patient.diabetes], 
        "cholesterol" : [patient.cholesterol], 
        "heart_rate" : [patient.heart_rate], 
        "chest_pain" : [patient.chest_pain], 
        "family_history" : [patient.family_history], 
        "bmi" : [patient.bmi], 
        "alcohol" : [patient.alcohol]
    })
    '''
    if index == 0:
        print('============================= HEALTHY')
    
    if index == 1289:
        print('============================= CORONARY')

    if index == 1621:
        print('============================= CEREBOVASCULAR')

    if index == 2012:
        print('============================= ARTERIAL')

    health_art_record = pd.DataFrame({
        "age" : [patient['age']],
        "sex" : [patient['sex']],
        "blood_pressure" : [patient['blood_pressure']],
        "cholesterol" : [patient['cholesterol']],
        "heart_rate" : [patient['heart_rate']]
    })

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

    health_cor_record = pd.DataFrame({
        "age" : [patient['age']], 
        "smoker" : [patient['smoker']], 
        "tobacco" : [patient['tobacco']], 
        "blood_pressure" : [patient['blood_pressure']], 
        "family_history" : [patient['family_history']], 
        "bmi" : [patient['bmi']], 
        "alcohol" : [patient['alcohol']]
    })

    art_cere_cor_record = pd.DataFrame({
        "age" : [patient['age']], 
        "sex" : [patient['sex']], 
        "blood_pressure" : [patient['blood_pressure']], 
        "cholesterol" : [patient['cholesterol']], 
        "heart_rate" : [patient['heart_rate']]
    })

    cere_cor_record = pd.DataFrame({
        "age" : [patient['age']], 
        "smoker" : [patient['smoker']], 
        "tobacco" : [patient['tobacco']], 
        "blood_pressure" : [patient['blood_pressure']], 
        "bmi" : [patient['bmi']]
    })

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

    #print(health_art_results)
    #print(health_cere_results)
    #print(health_cor_results)

    if sum(health_art_results) < 15 and sum(health_cere_results) < 8 and sum(health_cor_results) < 5:
        #print("You are predicted healthy")
        pred_list.append(0)
        count = 'healthy'

        if index < 1289:
            healthy_count += 1

    elif sum(health_art_results) >= 15 and sum(health_cere_results) < 8 and sum(health_cor_results) < 5:
        #print("The classifier has predicted that you may have arterial disease")
        pred_list.append(3)
        count = 'arterial'

        if index >= 2012 and index < 2165:
            art_count += 1

    elif sum(health_art_results) < 15 and sum(health_cere_results) >= 8 and sum(health_cor_results) < 5:
        #print("The classifier has predicted that you may have cerebovasular disease")
        pred_list.append(2)
        count = 'cerebovascular'

        if index >= 1621 and index < 2012:
            cere_count += 1

    elif sum(health_art_results) < 15 and sum(health_cere_results) < 8 and sum(health_cor_results) >= 5:
        #print("The classifier has predicted that you may have coronary heart disease")
        pred_list.append(1)
        count = 'coronary'

        if index >= 1289 and index < 1621:
            cor_count += 1

    elif sum(health_art_results) >= 15 and sum(health_cere_results) >= 8 and sum(health_cor_results) < 5:

        art_cere_results = []

        for key, value in art_cere_models.items():
            result = int(value.predict(art_cere_cor_record))
            art_cere_results.append(result)

        if art_cere_results.count(2) >= art_cere_results.count(3):
            pred_list.append(2)
            count = 'cerebovascular'

            if index >= 1621 and index < 2012:
                cere_count += 1

        else:
            pred_list.append(3)
            count = 'arterial'

            if index >= 2012 and index < 2165:
                art_count += 1

        #print("\n")
        #art_cere_results = art_cere_results.map(formatResultsArt)
        #print(art_cere_results)

    elif sum(health_art_results) < 15 and sum(health_cere_results) >= 8 and sum(health_cor_results) >= 5:

        cere_cor_results = []

        for key, value in cere_cor_models.items():
            result = int(value.predict(cere_cor_record))
            cere_cor_results.append(result)

        if cere_cor_results.count(2) >= cere_cor_results.count(1):
            pred_list.append(2)
            count = 'cerebovascular'

            if index >= 1621 and index < 2012:
                cere_count += 1

        else:
            pred_list.append(1)
            count = 'coronary'

            if index >= 1289 and index < 1621:
                cor_count += 1

        #print("\n")
        #cere_cor_results = cere_cor_results.map(formatResultsCor)
        #print(cere_cor_results)

    elif sum(health_art_results) >= 15 and sum(health_cere_results) < 8 and sum(health_cor_results) >= 5:


        art_cor_results = []

        for key, value in art_cor_models.items():
            result = int(value.predict(art_cere_cor_record))
            art_cor_results.append(result)

        if art_cor_results.count(3) >= art_cor_results.count(1):
            pred_list.append(3)
            count = 'arterial'

            if index >= 2012 and index < 2165:
                art_count += 1

        else:
            pred_list.append(1)
            count = 'coronary'

            if index >= 1289 and index < 1621:
                cor_count += 1

        #print("\n")
        #cere_cor_results = cere_cor_results.map(formatResultsCor)
        #print(art_cor_results)

    else:
        
        score = random.randint(1, 3)
        pred_list.append(score)
        count = 'multiple'

        if index >= 1289 and index < 1621:
            cor_count += 1

        elif index >= 1621 and index < 2012:
            cere_count += 1

        elif index >= 2012 and index < 2165:
            art_count += 1

    print(count)

#print(f'healthy: predicted = {pred_list.count(0)}')
#print(f'coronary: predicted = {pred_list.count(1)}')
#print(f'cerebovascular: predicted = {pred_list.count(2)}')
#print(f'arterial: predicted = {pred_list.count(3)}')

print("=========================")

health_score = healthy_count / 1288
print(health_score * 100)

cor_score = cor_count / 333
print(cor_score * 100)

cere_score = cere_count / 391
print(cere_score * 100)

art_score = art_count / 153
print(art_score * 100)

total_score = (healthy_count + cor_count + cere_count + art_count) / 2165
print(total_score * 100)

pred = np.array(pred_list)
real = combined_y.to_numpy()

'''
acc = accuracy_score(real, pred)
prec = precision_score(real, pred, pos_label=0)
rec = recall_score(real, pred, pos_label=0)
f1 = f1_score(real, pred, pos_label=0)
print(f"Accuracy score is {acc * 100}")
print(f"Precision score is {prec * 100}")
print(f"Recall score is {rec * 100}")
print(f"f1 score is {f1 * 100}")
'''

#real = {(combined_y == 1).sum()}

'''
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
'''