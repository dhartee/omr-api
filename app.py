from flask import Flask, request, jsonify
import cv2
import numpy as np
from omr_reader import process_custom_omr

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "OMR API is running successfully!"

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    # Image ko memory se read karein
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"error": "Failed to decode image"}), 400

    # OMR script ko image bhejein
    result = process_custom_omr(img)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
