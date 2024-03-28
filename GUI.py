import tkinter as tk
from tkinter import messagebox

def submit():
    messagebox.showinfo("Submitted", "Information submitted successfully!")

root = tk.Tk()
root.title("User Information")

labels = [
    "Age", 
    "Sex", 
    "Height", 
    "Weight", 
    "Diabetes (Yes/No)", 
    "Smoker (Yes/No)", 
    "Amount of Tobacco", 
    "Blood Pressure", 
    "Do you take any blood pressure medicine",  
    "Alcohol (Yes/No)",
    "Cholesterol",
    "Heart Rate",
    "Chest Pain (Yes/No)",
    "Family History (Yes/No)",
    "Blood Sugar (Yes/No)"]
entries = []

for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i, column=0)
    entries.append(tk.Entry(root))
    entries[-1].grid(row=i, column=1)

tk.Button(root, text="Submit", command=submit).grid(row=len(labels), column=1)


if __name__ == "__main__":
    root.mainloop()