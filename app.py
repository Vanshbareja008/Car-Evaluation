# app.py

import os
import joblib
import gradio as gr

# ==========================================================
# Load the trained model
# ==========================================================
# --- CODE BLOCK: LOAD XGBOOST MODEL ---
try:
    deployed_xgb = joblib.load("car_safety_model.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Model not found or error loading. {e}")
    deployed_xgb = None
# --------------------------------------

# ==========================================================
# Prediction Function with Bulletproof Error Handling
# ==========================================================
# --- CODE BLOCK: 6-FEATURE PREDICTION LOGIC ---
def predict_car_safety(
    buying_price,
    maintenance_cost,
    number_of_doors,
    number_of_persons,
    lug_boot,
    safety
):
    # Capture inputs from Gradio
    values = [
        buying_price, maintenance_cost, number_of_doors, 
        number_of_persons, lug_boot, safety
    ]

    # 1. Empty/None input check (Bulletproof catch if user skips a dropdown)
    if any(v is None or str(v).strip() == "" for v in values):
        return "❌ Please select an option for all input fields."

    # 2. Type casting to integers
    try:
        buying_price = int(buying_price)
        maintenance_cost = int(maintenance_cost)
        number_of_doors = int(number_of_doors)
        number_of_persons = int(number_of_persons)
        lug_boot = int(lug_boot)
        safety = int(safety)
    except (ValueError, TypeError):
        return "❌ Internal Error: Invalid data format received."

    # 3. Model execution
    if deployed_xgb is None:
        return "❌ Model failed to load. Please check your .pkl file."

    try:
        # Array strictly ordered to match the X dataframe provided
        input_data = [[
            buying_price,
            maintenance_cost,
            number_of_doors,
            number_of_persons,
            lug_boot,
            safety
        ]]

        prediction = deployed_xgb.predict(input_data)

        # Assuming standard label encoding for the 'decision' target
        # (Modify these return strings if your dataset used different target labels like unacc, acc, good, vgood)
        result_map = {
            0: "Unacceptable (unacc)",
            1: "Acceptable (acc)",
            2: "Good (good)",
            3: "Very Good (vgood)"
        }
        
        # Fallback to the raw prediction if it doesn't match 0-3
        final_decision = result_map.get(prediction[0], f"Class {prediction[0]}")

        return f"🚙 Evaluation Result\n\nCar Safety Decision: {final_decision}"

    except Exception as e:
        return f"❌ Prediction failed.\n\nError: {str(e)}"
# ----------------------------------------------

# ==========================================================
# Description & Footer
# ==========================================================
# --- CODE BLOCK: BRANDING & UI TEXT ---
DESCRIPTION = """
<div style="text-align:center;padding:15px;">

<h1 style="color:#2563EB;">
🚗 AI Car Safety Evaluation Dashboard
</h1>

<h3>
Powered by <span style="color:#16A34A;">XGBoost Machine Learning</span>
</h3>

<p style="font-size:17px;">
This intelligent dashboard evaluates a vehicle's overall
<b>acceptability and safety level</b> using a trained
Machine Learning model.
</p>

<hr>

<h3>📋 Required Vehicle Specifications</h3>

<table style="margin:auto;font-size:16px;">
<tr>
<td>💰 Buying Price</td>
<td>🛠 Maintenance Cost</td>
</tr>

<tr>
<td>🚪 Number of Doors</td>
<td>👨‍👩‍👧 Passenger Capacity</td>
</tr>

<tr>
<td>🧳 Luggage Boot Size</td>
<td>🛡 Safety Rating</td>
</tr>

</table>

<hr>

<p style="color:gray;">
Select all specifications and press
<b>🚀 Evaluate Vehicle</b> to generate the prediction report.
</p>

</div>
"""
developer_info = """
<hr>

<div style="padding:20px;border-radius:12px;background:#F8FAFC;">

<h2 align="center">👨‍💻 Developer Information</h2>

<table style="width:100%;font-size:16px;">

<tr>
<td><b>Developer</b></td>
<td>Vansh</td>
</tr>

<tr>
<td><b>Roll Number</b></td>
<td>241047</td>
</tr>

<tr>
<td><b>Machine Learning Model</b></td>
<td>XGBoost Classifier</td>
</tr>

<tr>
<td><b>Programming Language</b></td>
<td>Python</td>
</tr>

<tr>
<td><b>Framework</b></td>
<td>Gradio</td>
</tr>

<tr>
<td><b>Deployment Platform</b></td>
<td>Render Cloud</td>
</tr>

</table>

<hr>

<h3 align="center">🛠 Technology Stack</h3>

<div align="center">

Python • XGBoost • Joblib • Gradio • Render

</div>

<hr>

<div align="center">

<b>Version</b> 1.0.0

<br>

Designed for educational and machine learning demonstration purposes.

</div>

</div>
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="emerald",
    neutral_hue="slate",
).set(

    body_background_fill="#EEF4FF",

    block_background_fill="white",

    block_border_color="#CBD5E1",

    block_shadow="0px 4px 15px rgba(0,0,0,0.08)",

    button_primary_background_fill="#2563EB",

    button_primary_background_fill_hover="#1D4ED8",

    button_primary_text_color="white",

    input_background_fill="white",

    input_border_color="#CBD5E1",

    input_border_width="2px",

    radius_size="10px",

)




# --------------------------------------

# ==========================================================
# Interface Setup
# ==========================================================
# --- CODE BLOCK: SAFE DROPDOWN INPUTS ---
interface = gr.Interface(
    fn=predict_car_safety,
    inputs=[
        gr.Dropdown(
            choices=[("Low", 0), ("Medium", 1), ("High", 2), ("Very High", 3)],
            label="Buying Price"
        ),
        gr.Dropdown(
            choices=[("Low", 0), ("Medium", 1), ("High", 2), ("Very High", 3)],
            label="Maintenance Cost"
        ),
        gr.Dropdown(
            choices=[("2", 2), ("3", 3), ("4", 4), ("5 or More", 5)],
            label="Number of Doors"
        ),
        gr.Dropdown(
            choices=[("2", 2), ("4", 4), ("More", 5)],
            label="Number of Persons"
        ),
        gr.Dropdown(
            choices=[("Small", 0), ("Medium", 1), ("Big", 2)],
            label="Luggage Boot Size"
        ),
        gr.Dropdown(
            choices=[("Low", 0), ("Medium", 1), ("High", 2)],
            label="Safety Rating"
        ),
    ],
    outputs=gr.Textbox(label="Assessment Result", lines=4),
    title="🚙 Car Safety Evaluation System",
    description=DESCRIPTION,
    article=developer_info,
    theme=theme
)
# ----------------------------------------

# ==========================================================
# Launch Configuration
# ==========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Gradio server on 0.0.0.0:{port}...")
    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
