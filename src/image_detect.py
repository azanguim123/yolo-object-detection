from ultralytics import YOLO
import cv2
import os

class ImageDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image_path, save_output=True):
        results = self.model(image_path)

        # Récupérer image annotée
        annotated = results[0].plot()

        if save_output:
            os.makedirs("outputs", exist_ok=True)
            output_path = os.path.join("outputs", "result.jpg")
            cv2.imwrite(output_path, annotated)
            print(f"✅ Image sauvegardée : {output_path}")

        return annotated