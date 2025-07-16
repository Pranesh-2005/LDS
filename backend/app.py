from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
from PIL import Image
import numpy as np
import onnxruntime as ort
import os
import openai
from dotenv import load_dotenv  # Add this import

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)



class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


friendly_labels = {
    'Apple___Apple_scab': 'Apple scab',
    'Apple___Black_rot': 'Black rot',
    'Apple___Cedar_apple_rust': 'Cedar apple rust',
    'Apple___healthy': 'Healthy',
    'Blueberry___healthy': 'Healthy',
    'Cherry_(including_sour)___Powdery_mildew': 'Powdery mildew',
    'Cherry_(including_sour)___healthy': 'Healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Gray leaf spot',
    'Corn_(maize)___Common_rust_': 'Common rust',
    'Corn_(maize)___Northern_Leaf_Blight': 'Northern Leaf Blight',
    'Corn_(maize)___healthy': 'Healthy',
    'Grape___Black_rot': 'Black rot',
    'Grape___Esca_(Black_Measles)': 'Esca (Black Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Leaf blight',
    'Grape___healthy': 'Healthy',
    'Orange___Haunglongbing_(Citrus_greening)': 'Citrus greening',
    'Peach___Bacterial_spot': 'Bacterial spot',
    'Peach___healthy': 'Healthy',
    'Pepper,_bell___Bacterial_spot': 'Bacterial spot',
    'Pepper,_bell___healthy': 'Healthy',
    'Potato___Early_blight': 'Early blight',
    'Potato___Late_blight': 'Late blight',
    'Potato___healthy': 'Healthy',
    'Raspberry___healthy': 'Healthy',
    'Soybean___healthy': 'Healthy',
    'Squash___Powdery_mildew': 'Powdery mildew',
    'Strawberry___Leaf_scorch': 'Leaf scorch',
    'Strawberry___healthy': 'Healthy',
    'Tomato___Bacterial_spot': 'Bacterial spot',
    'Tomato___Early_blight': 'Early blight',
    'Tomato___Late_blight': 'Late blight',
    'Tomato___Leaf_Mold': 'Leaf mold',
    'Tomato___Septoria_leaf_spot': 'Leaf spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Spider mites',
    'Tomato___Target_Spot': 'Target spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Yellow Leaf Curl Virus',
    'Tomato___Tomato_mosaic_virus': 'Mosaic virus',
    'Tomato___healthy': 'Healthy'
}


# Set your Azure OpenAI credentials
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2024-12-01-preview"
openai.api_key = AZURE_OPENAI_KEY

def generate_description_and_prevention(disease_name):
    prompt = (
        f"Give a concise description (1-2 sentences) of the plant disease or condition in easy understandable language '{disease_name}', "
        "and a medium prevention method. Format the response as:\n"
        "Description: ...\nPrevention: ..."
    )
    client = openai.AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version="2024-12-01-preview",
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful plant pathology assistant and a expert in it with a huge ammount of knowladge."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.8,
    )
    content = response.choices[0].message.content
    description = "No description available."
    prevention = "No prevention info available."
    for line in content.split('\n'):
        if line.lower().startswith("description:"):
            description = line.split(":", 1)[1].strip()
        elif line.lower().startswith("prevention:"):
            prevention = line.split(":", 1)[1].strip()
    return description, prevention


def load_model():
    return ort.InferenceSession("leaf_model.onnx")

def preprocess_image(image, size=(224, 224)):
    image = image.resize(size)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

learn = load_model()


@app.route('/', methods=['POST'])
def index():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = file.filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        image_url = f"/static/uploads/{filename}"

        image = Image.open(save_path).convert("RGB")
        input_tensor = preprocess_image(image)
        input_name = learn.get_inputs()[0].name
        outputs = learn.run(None, {input_name: input_tensor})
        probs = outputs[0][0]
        pred_idx = np.argmax(probs)
        pred_class = class_names[pred_idx]
        friendly_pred = friendly_labels.get(pred_class, pred_class)
        confidence = probs[pred_idx] * 100
        description, prevention = generate_description_and_prevention(friendly_pred)
        return jsonify({
            'prediction': friendly_pred,
            'confidence': round(confidence, 2),
            'description': description,
            'prevention': prevention,
            'image_url': image_url
        })
    return jsonify({'error': 'Unknown error'}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)