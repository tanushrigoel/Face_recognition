import face_recognition

def extract_features(image_path):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    
    if face_encodings:
        return face_encodings[0]
    else:
        return None
