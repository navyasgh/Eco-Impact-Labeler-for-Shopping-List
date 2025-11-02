import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv('eco_foods.csv')
df_encoded = pd.get_dummies(df, columns=['category'])  
features = df_encoded.drop(['item', 'sustainable_alternative'], axis=1)  
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

knn = NearestNeighbors(n_neighbors=3, algorithm='auto')
knn.fit(features_scaled)

def calculate_impact(shopping_list):
   
    total_impact = 0
    impacts = {}
    unknown_items = []
    
    for item in shopping_list:
        item_lower = item.lower().strip()
        match = df[df['item'].str.lower() == item_lower]
        if not match.empty:
            footprint = match['carbon_footprint_kg_per_kg'].iloc[0]
            impacts[item] = footprint * 0.5  
            total_impact += impacts[item]
        else:
            unknown_items.append(item)
            impacts[item] = 0  
    
   
    if total_impact > 10:
        tip = "High impact! Aim for more plants to reduce by 50%."
    elif total_impact > 5:
        tip = "Moderate—swap meats for veggies."
    else:
        tip = "Eco-friendly list! Keep it up."
    
    return {
        'total_impact': round(total_impact, 2),
        'impacts': impacts,
        'tip': tip,
        'unknown_items': unknown_items
    }

def get_swaps(shopping_list):
    
    swaps = {}
    for item in shopping_list:
        item_lower = item.lower().strip()
        match_idx = df[df['item'].str.lower() == item_lower].index
        if len(match_idx) > 0:
            idx = match_idx[0]
            distances, indices = knn.kneighbors([features_scaled[idx]])
            similar_items = df.iloc[indices[0]]['sustainable_alternative'].tolist()
            swaps[item] = similar_items[0] if similar_items else "Local seasonal option" 
        else:
            swaps[item] = "Search for plant-based alternative"
    return swaps
