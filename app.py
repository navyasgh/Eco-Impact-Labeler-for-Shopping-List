
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import streamlit as st

data = [
    {"item": "chicken", "carbon_footprint_kg_per_kg": 6.9, "category": "meat", "sustainable_alternative": "lentils (legumes, nitrogen-fixing for soil)"},
    {"item": "salmon", "carbon_footprint_kg_per_kg": 20.0, "category": "fish", "sustainable_alternative": "mackerel (sustainable wild-caught)"},
    {"item": "tuna", "carbon_footprint_kg_per_kg": 6.1, "category": "fish", "sustainable_alternative": "sardines (low trophic level, abundant)"},
    {"item": "eggs", "carbon_footprint_kg_per_kg": 4.8, "category": "dairy", "sustainable_alternative": "fortified plant eggs (e.g., Just Egg, no animal farming)"},
    {"item": "milk", "carbon_footprint_kg_per_kg": 3.2, "category": "dairy", "sustainable_alternative": "oat milk (low water, renewable)"},
    {"item": "cheese", "carbon_footprint_kg_per_kg": 28.0, "category": "dairy", "sustainable_alternative": "almond cheese (nut-based, lower methane)"},
    {"item": "yogurt", "carbon_footprint_kg_per_kg": 2.2, "category": "dairy", "sustainable_alternative": "coconut yogurt (tropical, but local if possible)"},
    {"item": "butter", "carbon_footprint_kg_per_kg": 14.0, "category": "dairy", "sustainable_alternative": "avocado (plant fat, nutrient-dense)"},
    {"item": "rice", "carbon_footprint_kg_per_kg": 2.7, "category": "grains", "sustainable_alternative": "quinoa (ancient grain, drought-resistant)"},
    {"item": "wheat_bread", "carbon_footprint_kg_per_kg": 1.4, "category": "grains", "sustainable_alternative": "sourdough (fermented, reduces waste)"},
    {"item": "pasta", "carbon_footprint_kg_per_kg": 1.3, "category": "grains", "sustainable_alternative": "whole grain pasta (local wheat, fiber boost)"},
    {"item": "potatoes", "carbon_footprint_kg_per_kg": 0.3, "category": "vegetable", "sustainable_alternative": "sweet potatoes (similar, more vitamins)"},
    {"item": "carrots", "carbon_footprint_kg_per_kg": 0.2, "category": "vegetable", "sustainable_alternative": "beets (root veggies, seasonal)"},
    {"item": "spinach", "carbon_footprint_kg_per_kg": 0.4, "category": "vegetable", "sustainable_alternative": "kale (leafy greens, hardy crop)"},
    {"item": "broccoli", "carbon_footprint_kg_per_kg": 0.4, "category": "vegetable", "sustainable_alternative": "cauliflower (cruciferous, versatile)"},
    {"item": "tomatoes", "carbon_footprint_kg_per_kg": 1.4, "category": "vegetable", "sustainable_alternative": "cherry tomatoes (local greenhouse, less transport)"},
    {"item": "lettuce", "carbon_footprint_kg_per_kg": 0.3, "category": "vegetable", "sustainable_alternative": "arugula (fast-growing, pesticide-light)"},
    {"item": "apples", "carbon_footprint_kg_per_kg": 0.4, "category": "fruit", "sustainable_alternative": "local pears (tree fruit, in-season low transport)"},
    {"item": "bananas", "carbon_footprint_kg_per_kg": 0.8, "category": "fruit", "sustainable_alternative": "oranges (citrus, vitamin C alternative)"},
    {"item": "avocados", "carbon_footprint_kg_per_kg": 2.5, "category": "fruit", "sustainable_alternative": "seasonal berries (strawberries, lower water needs)"},
    {"item": "strawberries", "carbon_footprint_kg_per_kg": 0.4, "category": "fruit", "sustainable_alternative": "blueberries (antioxidants, pollinator-friendly)"},
    {"item": "oranges", "carbon_footprint_kg_per_kg": 0.4, "category": "fruit", "sustainable_alternative": "grapefruit (citrus family, similar freshness)"},
    {"item": "almonds", "carbon_footprint_kg_per_kg": 2.1, "category": "nuts", "sustainable_alternative": "peanuts (legume nut, less irrigation)"},
    {"item": "peanuts", "carbon_footprint_kg_per_kg": 0.8, "category": "nuts", "sustainable_alternative": "sunflower seeds (easy to grow, omega-3s)"},
    {"item": "chocolate", "carbon_footprint_kg_per_kg": 19.0, "category": "snack", "sustainable_alternative": "dark chocolate (cocoa fair-trade, minimal sugar)"},
    {"item": "cookies", "carbon_footprint_kg_per_kg": 3.5, "category": "snack", "sustainable_alternative": "oat bars (homemade, whole ingredients)"},
    {"item": "chips", "carbon_footprint_kg_per_kg": 2.0, "category": "snack", "sustainable_alternative": "popcorn (corn-based, air-popped low oil)"},
    {"item": "coffee", "carbon_footprint_kg_per_kg": 3.3, "category": "beverage", "sustainable_alternative": "tea (herbal, lower deforestation risk)"},
    {"item": "wine", "carbon_footprint_kg_per_kg": 2.5, "category": "beverage", "sustainable_alternative": "local milk (short supply chain)"},
    {"item": "soy_milk", "carbon_footprint_kg_per_kg": 0.8, "category": "dairy", "sustainable_alternative": "pea milk (innovative, low allergen)"},
    {"item": "tofu", "carbon_footprint_kg_per_kg": 2.0, "category": "protein", "sustainable_alternative": "edamame (whole soy, fresh protein)"},
    {"item": "quinoa", "carbon_footprint_kg_per_kg": 1.6, "category": "grains", "sustainable_alternative": "millet (gluten-free, water-efficient)"},
    {"item": "oats", "carbon_footprint_kg_per_kg": 0.9, "category": "grains", "sustainable_alternative": "barley (ancient, soil health benefits)"},
    {"item": "beans", "carbon_footprint_kg_per_kg": 0.8, "category": "protein", "sustainable_alternative": "peas (nitrogen-fixing, versatile)"},
    {"item": "lentils", "carbon_footprint_kg_per_kg": 0.9, "category": "protein", "sustainable_alternative": "chickpeas (hummus base, global staple)"},
    {"item": "mushrooms", "carbon_footprint_kg_per_kg": 0.4, "category": "vegetable", "sustainable_alternative": "shiitake (fungi, no tilling needed)"},
    {"item": "onions", "carbon_footprint_kg_per_kg": 0.4, "category": "vegetable", "sustainable_alternative": "garlic (alliums, flavor enhancers)"},
    {"item": "peppers", "carbon_footprint_kg_per_kg": 0.9, "category": "vegetable", "sustainable_alternative": "zucchini (summer squash, prolific yield)"},
    {"item": "cabbage", "carbon_footprint_kg_per_kg": 0.3, "category": "vegetable", "sustainable_alternative": "brussels sprouts (brassica, cold-hardy)"},
    {"item": "lemons", "carbon_footprint_kg_per_kg": 0.9, "category": "fruit", "sustainable_alternative": "limes (citrus, interchangeable in recipes)"},
    {"item": "honey", "carbon_footprint_kg_per_kg": 1.2, "category": "sweetener", "sustainable_alternative": "maple syrup (tree sap, sustainable tapping)"},
    {"item": "sugar", "carbon_footprint_kg_per_kg": 0.6, "category": "sweetener", "sustainable_alternative": "date syrup (fruit-based, natural)"},
    {"item": "olive_oil", "carbon_footprint_kg_per_kg": 3.2, "category": "oil", "sustainable_alternative": "canola oil (rapeseed, lower land use)"},
    {"item": "coconut_oil", "carbon_footprint_kg_per_kg": 1.5, "category": "oil", "sustainable_alternative": "sunflower oil (annual crop, high yield)"},
    {"item": "bread", "carbon_footprint_kg_per_kg": 1.4, "category": "grains", "sustainable_alternative": "flatbread (e.g., naan with local flour)"},
    {"item": "pizza", "carbon_footprint_kg_per_kg": 4.0, "category": "snack", "sustainable_alternative": "veggie wrap (tortilla with plants, customizable)"},
    {"item": "ice_cream", "carbon_footprint_kg_per_kg": 3.0, "category": "dairy", "sustainable_alternative": "sorbet (fruit-based, dairy-free)"}
]

df = pd.DataFrame(data)

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
        tip = "High impact! Aim for more plants to reduce by 50% "
    elif total_impact > 5:
        tip = "Moderate—swap meats for veggies. You're on the way!"
    else:
        tip = "Eco-friendly list! Keep it up—you're saving the planet! "
    
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
            neighbor_indices = [i for i in indices[0] if i != idx]
            if neighbor_indices:
                best_neighbor_idx = neighbor_indices[0]
                swaps[item] = df.iloc[best_neighbor_idx]['sustainable_alternative']
            else:
                swaps[item] = df.iloc[indices[0][0]]['sustainable_alternative']
        else:
            swaps[item] = "Search for plant-based alternative (e.g., via Open Food Facts API)"
    
    
    for item in swaps:
        swaps[item] += " (saves ~5-20kg CO2!)"
    
    return swaps

def main():
    st.title("Eco-Impact Labeler: Make Your Shopping Sustainable!") 
    st.write("Enter your grocery list (one item per line) and get a carbon score + green swaps!")

    shopping_list_input = st.text_area("Your Shopping List:", placeholder="e.g., chicken\nbanana\nrice", height=150)
    
    if st.button("Calculate Impact"):
        if shopping_list_input:
            shopping_list = [line.strip() for line in shopping_list_input.split('\n') if line.strip()]
            
           
            impact_result = calculate_impact(shopping_list)
            swaps = get_swaps(shopping_list)
            
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Carbon Footprint (kg CO2e)", impact_result['total_impact'])
                st.write("**Per-Item Breakdown:**")
                for item, score in impact_result['impacts'].items():
                    st.write(f"- **{item}**: {score} kg")
            
            with col2:
                st.write("**Sustainable Swaps:**")
                for item, swap in swaps.items():
                    st.write(f"- {item} → {swap}")
            
            eco_score = max(0, 100 - (impact_result['total_impact'] * 10))
            st.progress(eco_score / 100)
            st.success(f"Your Eco-Score: {eco_score:.0f}/100 ")
            st.info(impact_result['tip'])
            
            if impact_result['unknown_items']:
                st.warning(f"Unknown items: {', '.join(impact_result['unknown_items'])}. Add to dataset or use API!")
        
        else:
            st.warning("Add some items to your list!")

if __name__ == "__main__":
    main()
