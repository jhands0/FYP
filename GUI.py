import tkinter as tk
from tkinter import messagebox

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

    return



def submit():
    messagebox.showinfo("Submitted", "Information submitted successfully!")

root = tk.Tk()
root.title("Heart Disease Prediction and Classification")

labels = [
    "What is your age?", 
    "What is your sex?", 
    "What is your height?", 
    "What is your weight?", 
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

tk.Button(root, text="Submit", command=submit).grid(row=len(labels), column=1)


if __name__ == "__main__":
    root.mainloop()