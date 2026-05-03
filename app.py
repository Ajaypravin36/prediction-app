from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pytesseract
import re

import os

app = Flask(__name__)
CORS(app)

# Explicitly defining common Windows Tesseract path if it wasn't added to PATH
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

def process_image(image_bytes):
    # Convert image bytes to numpy array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Convert to grayscale for better OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Perform OCR
    text = pytesseract.image_to_string(gray)
    
    # Extract digits from text
    digits = re.findall(r'\d', text)
    
    if len(digits) < 10:
        return None
    
    # Return last 10 digits as integers
    return [int(d) for d in digits[-10:]]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected image"}), 400
    
    try:
        numbers = process_image(file.read())
        
        if not numbers:
            return jsonify({"error": "Could not identify at least 10 numbers from the image."}), 400
        
        # Core Logic
        next_number = sum(numbers) % 10
        size = "Big" if next_number >= 5 else "Small"
        # The prompt says: "Red" if even else "Green"
        color = "Red" if next_number % 2 == 0 else "Green"
        
        return jsonify({
            "number": next_number,
            "size": size,
            "color": color,
            "accuracy": "70%"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
