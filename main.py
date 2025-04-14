import cv2
import pytesseract
from keras.models import load_model
import numpy as np
# Update this to match your Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Load face and emotion detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('emotion_model.h5')  # Make sure you place this file in the same folder

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def extract_text(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text
def classify_question(text):
    if "?" in text and any(word in text.lower() for word in ["why", "explain", "describe"]):
        return "Descriptive"
    elif any(option in text for option in ["A)", "B)", "C)", "D)"]):
        return "MCQ"
    elif "fill in the blank" in text.lower():
        return "Fill-in-the-blank"
    else:
        return "Unknown"
def analyze_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        prediction = emotion_model.predict(roi)
        emotion = emotion_labels[np.argmax(prediction)]
        return emotion
    return "No face detected"
def run_learning_assistant(image_path, webcam=False):
    print("Reading question from image...")
    question = extract_text(image_path)
    print("Extracted Question:\n", question)

    question_type = classify_question(question)
    print("Question Type Detected:", question_type)

    if webcam:
        cap = cv2.VideoCapture(0)
        print("Analyzing facial expressions (press 'q' to quit)...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            emotion = analyze_emotion(frame)
            cv2.putText(frame, f'Emotion: {emotion}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Facial Emotion Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Run the assistant
run_learning_assistant("test.jpg", webcam=True)
            
       
    
