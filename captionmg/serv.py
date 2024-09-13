import os
import base64
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
import cv2
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
from gtts import gTTS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load pre-trained model, feature extractor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set generation parameters
gen_kwargs = {"max_length": 30, "num_beams": 10}

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.equalizeHist(gray)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_l = clahe.apply(l)
    enhanced_lab = cv2.merge((clahe_l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    pil_image = Image.fromarray(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    return pil_image


def predict_step(image_paths):
    images = [preprocess_image(image_path) for image_path in image_paths if preprocess_image(image_path)]
    if not images:
        return []
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [pred.strip() for pred in preds]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload_image')
def upload_image():
    return render_template('index.html')

@app.route('/capture_image')
def capture_image():
    return render_template('index1.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        predictions = predict_step([file_path])
        if predictions:
            caption = predictions[0]
            tts = gTTS(caption, lang='en')
            tts.save(os.path.join(app.config['UPLOAD_FOLDER'], 'caption.mp3'))
            return render_template('result.html', image_url=url_for('uploaded_file', filename=filename), caption=caption)
    return redirect(url_for('index'))

@app.route('/upload_capture', methods=['POST'])
def upload_capture():
    if 'image' not in request.form:
        return redirect(request.url)
    image_data = request.form['image'].split(",")[1]  # Remove the data URL scheme
    image_bytes = base64.b64decode(image_data)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.png')
    with open(image_path, 'wb') as f:
        f.write(image_bytes)
    predictions = predict_step([image_path])
    if predictions:
        caption = predictions[0]
        tts = gTTS(caption, lang='en')
        tts.save(os.path.join(app.config['UPLOAD_FOLDER'], 'caption.mp3'))
        return render_template('result.html', image_url=url_for('uploaded_file', filename='captured_image.png'), caption=caption)
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
