from flask import Flask, render_template, request, redirect, url_for
import os
from datetime import datetime
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLO model once
model = YOLO(r'Hackathon2_scripts/Hackathon2_scripts/runs/detect/train4/weights/best.pt')

# In-memory detection history
detection_history = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    
    # Save uploaded image
    filename = datetime.now().strftime("%Y%m%d%H%M%S_") + file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Load image with OpenCV
    image = cv2.imread(filepath)
    
    # Run YOLO inference
    results = model.predict(filepath)[0]  # YOLOv8
    boxes, class_names, confidences = [], [], []

    if results.boxes is not None:
        for i, box in enumerate(results.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(results.boxes.cls[i])
            conf = float(results.boxes.conf[i])
            label = model.names[cls_id]

            # Draw boxes
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            boxes.append([x1, y1, x2, y2])
            class_names.append(label)
            confidences.append(conf)
    
    # Save annotated image
    result_filename = "result_" + filename
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, image)

    
    detection_result = {
        'filename': result_filename,
        'detected_objects': class_names,
        'boxes': boxes,
        'confidences': confidences
    }
    
    # Add to history
    detection_history.append(detection_result)
    
    return render_template('result.html', results=detection_result, zip=zip)

@app.route('/history')
def history():
    return render_template('history.html', history=detection_history)

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/delete_history/<int:index>', methods=['POST'])
def delete_history(index):
    if 0 <= index < len(detection_history):
        detection_history.pop(index)
    return redirect(url_for('history'))

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)
