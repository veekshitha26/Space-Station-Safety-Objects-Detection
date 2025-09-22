# Space-Station-Safety-Objects-Detection
Duality AI Hackathon Project: Detecting safety objects in a Space Station using synthetic data from falcon and YOLO model.

Classes: OxygenTank, NitrogenTank, FirstAidBox, FireAlarm, SafetySwitchPanel, EmergencyPhone, FireExtinguisher.

Model Performence : The model achieved a precision of 87.39% and recall of 59.31% at epoch 10. The mAP@50 reached 69.00%, while mAP@50–95 achieved 57.73%.

<img width="471" height="270" alt="image" src="https://github.com/user-attachments/assets/d6a82f26-e57a-4237-ad05-3a12702783b8" />

## Web Interface : 
To make the object detection model user-friendly and accessible, a web-based interface was developed using Flask. 

Key Features:

- File Upload: Users can upload images in standard formats (.jpg, .png) for detection.
- Real-Time Detection: The uploaded image is processed by the trained YOLOv8 model, and predictions are generated immediately.
- Annotated Output: Detected objects are highlighted with bounding boxes, class labels, and confidence scores. The annotated images are displayed directly on the interface.
- History Option : Users can view the previously detected images and can also delete them.

<img width="464" height="195" alt="image" src="https://github.com/user-attachments/assets/ecc72bf8-68e5-43a8-a5c5-57895885fdaf" />



<img width="313" height="224" alt="image" src="https://github.com/user-attachments/assets/a27be441-5e79-4ad7-ade0-121723e4da5a" />



<img width="444" height="195" alt="image" src="https://github.com/user-attachments/assets/b8863496-d761-48d4-aa47-9d38205379df" />







                      

