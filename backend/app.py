import gradio as gr
from PIL import Image
import numpy as np
import onnxruntime as ort
import os
from dotenv import load_dotenv
import ast
from openai import OpenAI

# Load environment variables
load_dotenv()

# === Load and clean class names ===
class_file_path = "class_names.txt"
with open(class_file_path, "r") as f:
    raw_line = f.read()
class_names = ast.literal_eval(raw_line.replace("Classes: ", "").strip())

# === Load ONNX model ===
model_path = "model.onnx"
learn = ort.InferenceSession(model_path)

# === OpenRouter setup ===
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def generate_description_and_prevention(label):
    if label == "not_a_crop":
        return (
            "The uploaded image does not seem to show a valid crop or leaf.",
            "Please upload a clear image of a single crop or a leaf showing disease symptoms."
        )

    prompt = (
        f"Explain in simple words what the plant disease or condition '{label}' is, "
        f"and give 2 to 4 clear, practical prevention tips.\n"
        "Use this format:\n"
        "Description:\n"
        "Explain briefly what this disease is and how it affects the plant.\n"
        "Prevention:\n"
        "- Tip 1\n"
        "- Tip 2\n"
        "- (Optional) Tip 3\n"
        "- (Optional) Tip 4"
    )

    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-r1:free",
            messages=[
                {"role": "system", "content": "You are a knowledgeable plant pathologist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )

        content = response.choices[0].message.content
        if "Description:" in content and "Prevention:" in content:
            parts = content.split("Prevention:")
            description = parts[0].replace("Description:", "").strip()
            prevention = parts[1].strip()
            return description, prevention
        else:
            return "Description not structured correctly.", "No prevention steps found."
    except Exception as e:
        print(f"[ERROR] OpenRouter API error: {e}")
        return "OpenRouter error.", "Failed to generate prevention steps."

def preprocess_image(image, size=(224, 224)):
    image = image.resize(size)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image):
    image = image.convert("RGB")
    input_tensor = preprocess_image(image)

    input_name = learn.get_inputs()[0].name
    outputs = learn.run(None, {input_name: input_tensor})
    probs = outputs[0][0]
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx] * 100)

    description, prevention = generate_description_and_prevention(pred_class)

    return pred_class, round(confidence, 2), description, prevention

# === Gradio Interface ===
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Number(label="Confidence %"),
        gr.Textbox(label="Description"),
        gr.Textbox(label="Prevention")
    ],
    title="🌱 Crop Disease Detection",
    description="Upload a crop or leaf image to detect plant diseases and get prevention tips."
)

if __name__ == "__main__":
    iface.launch(debug=True)