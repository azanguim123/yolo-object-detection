import cv2
import time
from ultralytics import YOLO
from utils import filter_classes, count_objects

class WebcamDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.target_classes = ["person", "car"]

    def run(self):
        cap = cv2.VideoCapture(0)

        
        # Intitialisation video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('outputs/output.mp4', fourcc, 20.0, (640, 480))
        prev_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)

            # 🔍 Filtrage
            filtered = filter_classes(results, self.target_classes)

            # 📊 Comptage
            counts = count_objects(filtered)

            # 🖼️ Image annotée
            annotated_frame = results[0].plot()

            # ⚡ FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            # 📝 Affichage infos
            y_offset = 30
            for label, count in counts.items():
                text = f"{label}: {count}"
                cv2.putText(annotated_frame, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                y_offset += 30

            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.imshow("YOLO Advanced Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        # Liberation
        cap.release()
        out.release() 
        cv2.destroyAllWindows()  