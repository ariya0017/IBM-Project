import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model and tokenizer
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

# Fix pad_token issue
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Use pipeline for easier inference
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # device=0 if torch.cuda.is_available() else -1 # Removed the device argument
)

def generate_response(prompt, max_length=512):
    outputs = generator(
        prompt,
        max_new_tokens=max_length,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = outputs[0]["generated_text"]
    # remove echo if exists
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    return response

def disease_prediction(symptoms):
    prompt = f"""You are a medical assistant.
Based on the following symptoms, provide possible medical conditions and general medication suggestions.
Always emphasize the importance of consulting a doctor for proper diagnosis.

Symptoms: {symptoms}

Possible conditions and recommendations:

**IMPORTANT: This is for informational purposes only. Please consult a healthcare professional for proper diagnosis and treatment.**

Analysis:"""
    return generate_response(prompt, max_length=400)

def treatment_plan(condition, age, gender, medical_history):
    prompt = f"""You are a medical assistant.
Generate personalized treatment suggestions for the following patient information.
Include home remedies and general medication guidelines.

Medical Condition: {condition}
Age: {age}
Gender: {gender}
Medical History: {medical_history}

Personalized treatment plan including home remedies and medication guidelines:

**IMPORTANT: This is for informational purposes only. Please consult a healthcare professional for proper treatment.**

Treatment Plan:"""
    return generate_response(prompt, max_length=400)

# Build Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# 🏥 Medical AI Assistant")
    gr.Markdown("⚠️ **Disclaimer:** This tool is for informational purposes only. Always consult a doctor for medical advice.")

    with gr.Tabs():
        with gr.TabItem("Disease Prediction"):
            with gr.Row():
                with gr.Column():
                    symptoms_input = gr.Textbox(
                        label="Enter Symptoms",
                        placeholder="e.g., fever, headache, cough, fatigue...",
                        lines=4
                    )
                    predict_btn = gr.Button("Analyze Symptoms")

                with gr.Column():
                    prediction_output = gr.Textbox(label="Possible Conditions & Recommendations", lines=20)

            predict_btn.click(disease_prediction, inputs=symptoms_input, outputs=prediction_output)

        with gr.TabItem("Treatment Plans"):
            with gr.Row():
                with gr.Column():
                    condition_input = gr.Textbox(
                        label="Medical Condition",
                        placeholder="e.g., diabetes, hypertension, migraine...",
                        lines=2
                    )
                    age_input = gr.Number(label="Age", value=30)
                    gender_input = gr.Dropdown(
                        choices=["Male", "Female", "Other"],
                        label="Gender",
                        value="Male"
                    )
                    history_input = gr.Textbox(
                        label="Medical History",
                        placeholder="Previous conditions, allergies, medications or None",
                        lines=3
                    )
                    plan_btn = gr.Button("Generate Treatment Plan")

                with gr.Column():
                    plan_output = gr.Textbox(label="Personalized Treatment Plan", lines=20)

            plan_btn.click(
                treatment_plan,
                inputs=[condition_input, age_input, gender_input, history_input],
                outputs=plan_output
            )

app.launch(share=True)
