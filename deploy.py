import streamlit as st
import pickle
import pandas as pd

# Load the trained model and scaler
model = pickle.load(open("Recommendation_model.pkl", 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))

# Load the dataset for reference
df = pd.read_csv("clean_data.csv") 

# # Recommendation function
def recommend_neighborhoods(price):
    price_scaled = scaler.transform([[price]])
    distances, indices = model.kneighbors(price_scaled)
    return df.loc[indices[0], ["Address", "Neighborhood", "Bedrooms", "Property_Type", "Price"]].reset_index(drop=True)

# Streamlit UI
st.title("Lagos House Price Neighborhood Recommender")

user_price = st.number_input("Enter your budget (₦):", min_value=100000, step=50000)

if st.button("Find Recommendations"):
    recommendations = recommend_neighborhoods(user_price)
    st.write(recommendations)

