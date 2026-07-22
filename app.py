import os
import joblib
import pandas as pd
import gradio as gr

# ==========================================================
# Load Model
# ==========================================================
model = joblib.load("car_safety_model.pkl")

# ==========================================================
# Prediction Function
# ==========================================================
def predict_car_safety(
    buying,
    maintenance,
    doors,
    persons,
    lug_boot,
    safety
):

    # DataFrame with EXACT feature names used while training
    data = pd.DataFrame([{
        "buying price": buying,
        "maintenance cost": maintenance,
        "number of doors": doors,
        "number of persons": persons,
        "lug_boot": lug_boot,
        "safety": safety
    }])

    prediction = model.predict(data)[0]

    try:
        confidence = max(model.predict_proba(data)[0]) * 100
        return f"Prediction : {prediction}\nConfidence : {confidence:.2f}%"
    except:
        return f"Prediction : {prediction}"


# ==========================================================
# Gradio Interface
# ==========================================================

title = """
# 🚗 Car Safety Prediction System

### Developed by **Vansh**
### Roll No. **241047**
"""

description = """
Predict the **Safety Category** of a car based on its specifications.

### Example Values
Buying Price : low

Maintenance : low

Doors : 4

Persons : 4

Luggage Boot : big

Safety : high
"""

examples = [
    ["low", "low", "4", "4", "big", "high"],
    ["high", "high", "2", "2", "small", "low"],
    ["med", "med", "4", "more", "med", "med"],
    ["vhigh", "vhigh", "2", "2", "small", "low"]
]

demo = gr.Interface(
    fn=predict_car_safety,

    inputs=[
        gr.Dropdown(
            ["low", "med", "high", "vhigh"],
            label="Buying Price"
        ),

        gr.Dropdown(
            ["low", "med", "high", "vhigh"],
            label="Maintenance Cost"
        ),

        gr.Dropdown(
            ["2", "3", "4", "5more"],
            label="Number of Doors"
        ),

        gr.Dropdown(
            ["2", "4", "more"],
            label="Number of Persons"
        ),

        gr.Dropdown(
            ["small", "med", "big"],
            label="Luggage Boot Size"
        ),

        gr.Dropdown(
            ["low", "med", "high"],
            label="Safety"
        )
    ],

    outputs=gr.Textbox(label="Prediction"),

    title=title,

    description=description,

    examples=examples,

    theme=gr.themes.Soft()
)

# ==========================================================
# Required for Render Deployment
# ==========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
