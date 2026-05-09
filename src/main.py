from image_detect import ImageDetector
from webcam import WebcamDetector

def main():
    print("=== YOLO Object Detection ===")
    print("1. Image Detection")
    print("2. Webcam Detection")

    choice = input("Choisir une option (1 ou 2): ")

    if choice == "1":
        image_path = input("Chemin de l'image: ")
        detector = ImageDetector()
        detector.detect(image_path)

    elif choice == "2":
        detector = WebcamDetector()
        detector.run()

    else:
        print("❌ Choix invalide")

if __name__ == "__main__":
    main()