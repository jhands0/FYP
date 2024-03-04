from ucimlrepo import fetch_ucirepo 

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
  
combined = X.join(y)

print(combined['num'].value_counts())

combined.to_csv("datasets/HeartDisease.csv")