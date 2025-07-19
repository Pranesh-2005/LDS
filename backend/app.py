from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import onnxruntime as ort
import os
import openai
from dotenv import load_dotenv
import threading
import time
import ast
import sys

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# === Load and clean class names ===
try:
    class_file_path = os.path.join(os.path.dirname(__file__), 'class_names.txt')
    with open(class_file_path, 'r') as f:
        raw_line = f.read()
    class_names = ast.literal_eval(raw_line.replace("Classes: ", "").strip())

    # Optional: rewrite cleaned names
    with open(class_file_path, 'w') as f:
        for cls in class_names:
            f.write(cls + '\n')

except FileNotFoundError:
    print("[ERROR] class_names.txt not found. Make sure it's present in the backend directory.")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Failed to load class names: {e}")
    sys.exit(1)

# === Load ONNX model ===
try:
    model_path = os.path.join(os.path.dirname(__file__), 'model.onnx')
    learn = ort.InferenceSession(model_path)
except Exception as e:
    print(f"[ERROR] Failed to load ONNX model: {e}")
    sys.exit(1)

# === Azure OpenAI setup ===
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2024-12-01-preview"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def generate_description_and_prevention(label):
    if label == "not_a_crop":
        return (
            "The uploaded image does not seem to depict a recognizable fruit or leaf.",
            "Please ensure the image clearly shows a single crop, preferably a close-up of a fruit or diseased leaf."
        )

    prompt = (
        f"Provide a concise, clear description and point-wise prevention methods for the following plant disease or condition: '{label}'.\n"
        "Format:\n"
        "Description:\n"
        "- ...\n"
        "Prevention:\n"
        "- ... (2-4 bullet points)"
    )

    try:
        client = openai.AzureOpenAI(
            api_key=openai.api_key,
            api_version=openai.api_version,
            azure_endpoint=openai.api_base
        )

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a knowledgeable plant pathologist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
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
        print(f"[ERROR] OpenAI API error: {e}")
        return "OpenAI error.", "Failed to generate prevention steps."

def preprocess_image(image, size=(224, 224)):
    image = image.resize(size)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def cleanup_uploads(folder, lifetime=3600):
    while True:
        now = time.time()
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) and now - os.path.getmtime(file_path) > lifetime:
                try:
                    os.remove(file_path)
                    print(f"[CLEANUP] Deleted: {file_path}")
                except Exception as e:
                    print(f"[CLEANUP ERROR] Failed to delete {file_path}: {e}")
        time.sleep(600)

@app.route('/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = file.filename
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    image_url = f"/static/uploads/{filename}"

    image = Image.open(save_path).convert("RGB")
    input_tensor = preprocess_image(image)

    input_name = learn.get_inputs()[0].name
    outputs = learn.run(None, {input_name: input_tensor})
    probs = outputs[0][0]
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx] * 100)

    description, prevention = generate_description_and_prevention(pred_class)

    return jsonify({
        'prediction': pred_class,
        'confidence': round(confidence, 2),
        'description': description,
        'prevention': prevention,
        'image_url': image_url
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    cleanup_thread = threading.Thread(target=cleanup_uploads, args=(app.config['UPLOAD_FOLDER'],), daemon=True)
    cleanup_thread.start()
    app.run(host="0.0.0.0", port=port)
