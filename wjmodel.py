import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pickle
import sys

current_dir = Path(__file__).parent.resolve()
model_path = current_dir / "XGBoost.pkl"
with open('XGBoost.pkl','rb') as file:
    model=pickle.load(file)

feature_names = ['D_Dimer','AST','Cl','Cough_Dur', 'Ca','LDH','MONO','CR']
st.title("Predicting the severity of mycoplasma pneumoniae pneumonia")
st.write('Please enter the following clinical indicators to predict the severity of mycoplasma pneumoniae pneumonia:')
input_D_Dimer = st.number_input("D-Dimer(D-Dimer mg/L):", min_value=0.0, max_value=1.0, value=0.5, format="%.2f")
input_AST = st.number_input("AST(aspartate aminotransferase U/L):", min_value=0.0, max_value=1200.0, value=20.00, format="%.2f")
input_Cl = st.number_input("Cl(Chloride mmol/L):", min_value=50.0, max_value=150.0, value=80.00, format="%.2f")
input_Cough_Dur = st.number_input("Cough_Dur(Duration of cough days):", min_value=0.0, max_value=30.0, value=5.00, format="%.1f")
input_LDH = st.number_input("LDH(lactate dehydrogenase U/L):", min_value=100.0, max_value=500.0, value=300.00, format="%.2f")
input_Ca = st.number_input("Ca(Calcium mmol/L):", min_value=1.0, max_value=5.0, value=2.00, format="%.2f")
input_MONO = st.number_input("MONO(Monocyte count 10^9/L):", min_value=0.01, max_value=1.50, value=0.50, format="%.2f")
input_CR = st.number_input("CR(Creatinine μmol/L):", min_value=10.00, max_value=60.00, value=33.00, format="%.2f")
feature_values = [
    input_Cough_Dur,input_D_Dimer,input_AST,input_Cl,input_LDH,input_Ca,input_MONO,input_CR]
features = np.array([feature_values])
if st.button("Predict"):
   Predicted_Degree = model.predict(features)[0]
   predicted_proba = model.predict_proba(features)[0]
    
   probability = predicted_proba[Predicted_Degree] * 100
   if Predicted_Degree == 1:
         st.write(f"Predicted Degree Severe")
         st.write(f"**Prediction Probabilities:** {predicted_proba}")
         st.write( f"According to our model, you have a high risk of severity mycoplasma pneumoniae pneumonia(SMMP). ")
         st.write(f"The model predicts that your probability of having SMMP disease is {probability:.1f}%. ")
   else:
        st.write(f"Predicted Degree Mild")
        st.write(f"**Prediction Probabilities:** {predicted_proba}")
        st.write( f"According to our model, you have a low risk of severity mycoplasma pneumoniae pneumonia(SMMP). ")
        st.write( f"The model predicts that your probability of not having SMMP disease is {probability:.1f}%. ")

   explainer = shap.TreeExplainer(model)
   shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

   shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
   plt.savefig("shap_plot.png", bbox_inches='tight', dpi=1200)

   st.image("shap_plot.png")

   

