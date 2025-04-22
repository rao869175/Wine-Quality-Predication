# Wine Quality Prediction with Gradio
import os
os.system("pip install scikit-learn")


import pandas as pd
import gradio as gr
import pickle

# Load trained classifier model from pickle
with open('wine_quality_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Define feature names
feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']

# Prediction function for Gradio
def predict_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                    chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol):
    input_data = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]],
                                columns=feature_names)
    prediction = model.predict(input_data)[0]
    return f"Predicted Wine Quality: {prediction}"

# Gradio Inputs
inputs = [
    gr.Number(label="Fixed Acidity"),
    gr.Number(label="Volatile Acidity"),
    gr.Number(label="Citric Acid"),
    gr.Number(label="Residual Sugar"),
    gr.Number(label="Chlorides"),
    gr.Number(label="Free Sulfur Dioxide"),
    gr.Number(label="Total Sulfur Dioxide"),
    gr.Number(label="Density"),
    gr.Number(label="pH"),
    gr.Number(label="Sulphates"),
    gr.Number(label="Alcohol")
]

# Gradio Interface
interface = gr.Interface(
    fn=predict_quality,
    inputs=inputs,
    outputs="text",
    title="🍷 Wine Quality Predication",
    description="Enter wine chemical properties to predict the wine quality rating (3–8)."
)

# Launch app
interface.launch()

