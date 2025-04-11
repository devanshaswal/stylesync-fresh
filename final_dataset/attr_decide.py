import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold

file_path = "data/processed/final_merged/final_normalized.csv"
df = pd.read_csv(file_path)


attribute_columns = df.columns[1:1001]  


print(f"Total attributes: {len(attribute_columns)}")
print(f"Sample attributes: {attribute_columns[:10]}")



X = df.iloc[:, 1:1001]  

selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)
selected_attributes = X.columns[selector.get_support()]

print(f"Remaining attributes after variance filtering: {len(selected_attributes)}")

if "category_label" in df.columns:
    y = df["category_label"]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df[selected_attributes], y)
    
    importances = model.feature_importances_
    
    top_n = 50  
    top_indices = np.argsort(importances)[-top_n:]  
    top_attributes = selected_attributes[top_indices]  

    print(f"Top {top_n} attributes:\n", top_attributes)
