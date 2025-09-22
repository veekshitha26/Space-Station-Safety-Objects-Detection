from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml

# Function to predict and save images
def predict_and_save(model, image_path, output_path, output_path_txt):
    results = model.predict(image_path, conf=0.5)
    result = results[0]
    img = result.plot()  
    cv2.imwrite(str(output_path), img)

    with open(output_path_txt, 'w') as f:
        for box in result.boxes:
            cls_id = int(box.cls)
            x_center, y_center, width, height = box.xywhn[0].tolist()
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")


if __name__ == '__main__': 

    this_dir = Path(__file__).parent
    os.chdir(this_dir)

    # Load YAML parameters
    with open(this_dir / 'yolo_params.yaml', 'r') as file:
        data = yaml.safe_load(file)
        if 'test' in data and data['test'] is not None:
            images_dir = Path(data['test'])
        else:
            print("No test field found in yolo_params.yaml, please add the test field with the path to the test images")
            exit()

    print(f"Images directory: {images_dir}")
    print(f"Exists? {images_dir.exists()}")

    # Validate images directory
    if not images_dir.exists():
        print(f"Images directory {images_dir} does not exist")
        exit()
    if not images_dir.is_dir():
        print(f"Images directory {images_dir} is not a directory")
        exit()
    if not any(images_dir.iterdir()):
        print(f"Images directory {images_dir} is empty")
        exit()

    # Load the YOLO model
    detect_path = this_dir / "runs" / "detect"
    train_folders = [f for f in os.listdir(detect_path) if os.path.isdir(detect_path / f) and f.startswith("train")]
    if len(train_folders) == 0:
        raise ValueError("No training folders found")

    # Pick the first training folder if only one
    idx = 0
    if len(train_folders) > 1:
        choice = -1
        choices = list(range(len(train_folders)))
        while choice not in choices:
            print("Select the training folder:")
            for i, folder in enumerate(train_folders):
                print(f"{i}: {folder}")
            choice = input()
            if not choice.isdigit():
                choice = -1
            else:
                choice = int(choice)
        idx = choice

    model_path = detect_path / train_folders[idx] / "weights" / "best.pt"
    model = YOLO(model_path)

    # Output directories
    output_dir = this_dir / "predictions"
    images_output_dir = output_dir / 'images'
    labels_output_dir = output_dir / 'labels'
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    # Predict on all images
    for img_path in images_dir.glob('*'):
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
        output_path_img = images_output_dir / img_path.name
        output_path_txt = labels_output_dir / img_path.with_suffix('.txt').name
        predict_and_save(model, img_path, output_path_img, output_path_txt)

    print(f"Predicted images saved in {images_output_dir}")
    print(f"Bounding box labels saved in {labels_output_dir}")

    metrics = model.val(data=this_dir / 'yolo_params.yaml', split="test")
