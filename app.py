from flask import Flask, request, jsonify
import cv2
import numpy as np
import fitz  # PyMuPDF
from omr_reader import process_custom_omr

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "OMR API is running successfully!"

@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    filename = file.filename.lower()
    
    try:
        if filename.endswith('.pdf'):
            # PDF ko image mein convert karein
            pdf_bytes = file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page = doc.load_page(0) # First page
            pix = page.get_pixmap(dpi=150)
            
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        else:
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            
        if img is None:
            return jsonify({"error": "Failed to decode file"}), 400

        result = process_custom_omr(img)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
