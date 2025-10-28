from ultralytics import YOLO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import cv2
import numpy as np
import os

app = Flask(__name__)

# Update CORS to allow your Vercel frontend
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",  # Local development
            "http://localhost:5173",  # Vite local dev
            "https://*.vercel.app",   # Your Vercel deployments
            # Add your custom domain here when ready
        ]
    }
})

# Lazy load YOLO model
model = None

def get_model():
    global model
    if model is None:
        model = YOLO('yolov8n.pt')
    return model

@app.route('/detect', methods=['POST'])
def detect_objects():
    try:
        # Get image from request
        data = request.json
        image_data = data['image'].split(',')[1]  # Remove data:image/png;base64, prefix
        target_object = data.get('target', None)  # Optional target object
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run YOLO detection
        results = get_model().predict(image)
        
        # Extract detection data
        detections = []
        target_found = False
        target_confidence = 0
        
        for r in results:
            for box in r.boxes:
                class_name = r.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': box.xyxy[0].tolist()
                })
                
                # Check if target object was found
                if target_object and class_name.lower() == target_object.lower():
                    target_found = True
                    target_confidence = max(target_confidence, confidence)
        
        response = {
            'success': True,
            'detections': detections
        }
        
        # Add target-specific info if target was provided
        if target_object:
            response['target_found'] = target_found
            response['target_confidence'] = target_confidence
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/detect-debug', methods=['POST'])
def detect_objects_debug():
    try:
        # Get image from request
        data = request.json
        image_data = data['image'].split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run YOLO detection with visualization
        results = get_model().predict(image)
        
        # Extract detection data
        detections = []
        for r in results:
            for box in r.boxes:
                class_name = r.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        # Get annotated image
        annotated_image = results[0].plot()
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', annotated_image)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'detections': detections,
            'annotated_image': f'data:image/png;base64,{annotated_base64}'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)