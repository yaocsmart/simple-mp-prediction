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
model_path = r"F:\\WJblood\WJsteamlit\XGBoost.pkl"
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model=pickle.load(f)
model = load_model('XGBoost.pkl')

st.title("Predicting the severity of mycoplasma pneumoniae pneumonia")
feature_names = ['D_Dimer','AST','Cl','Cough_Dur', 'Ca','LDH','MONO','CR']
st.write('Please enter the following clinical indicators to predict the severity of mycoplasma pneumoniae pneumonia:')
D_Dimer = st.number_input("D-Dimer(D-Dimer mg/L):", min_value=0.0, max_value=1.0, value=0.5, format="%.2f")
AST = st.number_input("AST(aspartate aminotransferase U/L):", min_value=0.0, max_value=1200.0, value=20.00, format="%.2f")
Cl = st.number_input("Cl(Chloride mmol/L):", min_value=50.0, max_value=150.0, value=80.00, format="%.2f")
Cough_Dur = st.number_input("Cough_Dur(Duration of cough days):", min_value=0.0, max_value=30.0, value=5.00, format="%.1f")
LDH = st.number_input("LDH(lactate dehydrogenase U/L):", min_value=100.0, max_value=500.0, value=300.00, format="%.2f")
Ca = st.number_input("Ca(Calcium mmol/L):", min_value=1.0, max_value=5.0, value=2.00, format="%.2f")
MONO = st.number_input("MONO(Monocyte count 10^9/L):", min_value=0.01, max_value=1.50, value=0.50, format="%.2f")
CR = st.number_input("CR(Creatinine μmol/L):", min_value=10.00, max_value=60.00, value=33.00, format="%.2f")
feature_values = [Cough_Dur,D_Dimer,AST,Cl,LDH,Ca,MONO,CR]
features = np.array([feature_values], dtype=np.float64)

if st.button("Predict"):
    if model is None:
        st.error("❌ Prediction failed: Model not loaded, please check the XGBoost.pkl file")
    elif not hasattr(model, "predict"):
        st.error("❌ Invalid model: Loaded object is not a valid sklearn model")
    else:
        try:
            # Core prediction
            Predicted_Degree = model.predict(features)[0]
            predicted_proba = model.predict_proba(features)[0]
            probability = predicted_proba[Predicted_Degree] * 100

            # Display results
            if Predicted_Degree == 1:
                st.success(f"✅ Prediction Result: Severe")
                st.write(f"**Severe Probability: {probability:.1f}%**")
            else:
                st.success(f"✅ Prediction Result: Mild")
                st.write(f"**Mild Probability: {probability:.1f}%**")

            st.write(f"Full Probability (Mild/Severe): {np.round(predicted_proba, 4)}")

            # ====================== 4. SHAP Visualization ======================
            try:
                explainer = shap.TreeExplainer(model)
                df_input = pd.DataFrame([feature_values], columns=feature_names)
                shap_values = explainer.shap_values(df_input)

                if isinstance(shap_values, list):
                    shap_val = shap_values[1][0]
                    base_val = explainer.expected_value[1]
                else:
                    shap_val = shap_values[0]
                    base_val = explainer.expected_value

                shap.force_plot(
                    base_val,
                    shap_val,
                    df_input,
                    matplotlib=True,
                    show=False
                )
                plt.tight_layout()
                plt.savefig("shap_force_plot.png", bbox_inches="tight", dpi=300)
                plt.close()
                st.image("shap_force_plot.png", caption="SHAP Feature Contribution Explanation")
            except Exception as shap_err:
                st.warning(f"⚠️ SHAP plot generation failed: {str(shap_err)[:100]}")

        except Exception as pred_err:
            st.error(f"❌ Prediction failed: {str(pred_err)}")
            st.info(f"Debug info: Model type={type(model)}, Input shape={features.shape}")
   

