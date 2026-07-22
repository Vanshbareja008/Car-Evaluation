import gradio as gr
import joblib
import pandas as pd

# =====================================================
# Load Model
# =====================================================
try:
    model = joblib.load("car_safety_model.pkl")
except Exception as e:
    model = None
    print("Model Loading Error:", e)

# =====================================================
# Prediction Function
# =====================================================
def predict_car_safety(
    buying_price,
    maintenance_cost,
    number_of_doors,
    number_of_persons,
    lug_boot,
    safety
):
    if model is None:
        return "❌ Model not loaded.", ""

    # Create dataframe with exact feature names
    input_data = pd.DataFrame([{
        "buying price": buying_price,
        "maintenance cost": maintenance_cost,
        "number of doors": number_of_doors,
        "number of persons": number_of_persons,
        "lug_boot": lug_boot,
        "safety": safety
    }])

    prediction = model.predict(input_data)[0]

    confidence = ""

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        confidence = f"Confidence : {max(probs)*100:.2f}%"

    return str(prediction), confidence


# =====================================================
# Gradio Interface
# =====================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
# 🚗 Car Safety Prediction System

### Predict the Safety Category of a Car using Machine Learning

**Developed by : Vansh**  
**Roll No. : 241047**

---
"""
    )

    with gr.Row():

        buying_price = gr.Dropdown(
            ["low", "med", "high", "vhigh"],
            label="Buying Price",
            value="low"
        )

        maintenance_cost = gr.Dropdown(
            ["low", "med", "high", "vhigh"],
            label="Maintenance Cost",
            value="low"
        )

    with gr.Row():

        number_of_doors = gr.Dropdown(
            ["2", "3", "4", "5more"],
            label="Number of Doors",
            value="4"
        )

        number_of_persons = gr.Dropdown(
            ["2", "4", "more"],
            label="Capacity (Persons)",
            value="4"
        )

    with gr.Row():

        lug_boot = gr.Dropdown(
            ["small", "med", "big"],
            label="Luggage Boot Size",
            value="med"
        )

        safety = gr.Dropdown(
            ["low", "med", "high"],
            label="Safety",
            value="high"
        )

    predict_btn = gr.Button("🔍 Predict Safety", variant="primary")

    output = gr.Textbox(label="Prediction")
    confidence = gr.Textbox(label="Model Confidence")

    predict_btn.click(
        predict_car_safety,
        inputs=[
            buying_price,
            maintenance_cost,
            number_of_doors,
            number_of_persons,
            lug_boot,
            safety
        ],
        outputs=[
            output,
            confidence
        ]
    )

    gr.Markdown(
        """
## 📌 Example Values

| Feature | Example |
|---------|---------|
| Buying Price | low |
| Maintenance Cost | low |
| Number of Doors | 4 |
| Capacity | 4 |
| Luggage Boot | med |
| Safety | high |

Click **Predict Safety** to view the predicted class.
"""
    )

# =====================================================
# Launch
# =====================================================
import os
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
