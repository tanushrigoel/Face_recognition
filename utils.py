import cv2
import os

def capture_images(name):
    os.makedirs(f"dataset/{name}", exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow("Capturing Images", frame)
        
        cv2.imwrite(f"dataset/{name}/{count}.jpg", frame)
        count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
