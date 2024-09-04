from face_detection import detect_faces
from feature_extraction import extract_features
from face_recognition import recognize_faces
from real_time_recognition import recognize_faces_in_video
from utils import capture_images

def main():
    # Example usage
    name = input("Enter your name: ")
    capture_images(name)
    
    # Load known faces
    # ... (add your code here to load known faces)
    
    # Test recognition
    image_path = "dataset/your_name/0.jpg"
    recognize_faces(image_path)
    
    # Real-time recognition
    recognize_faces_in_video()

if __name__ == "__main__":
    main()
