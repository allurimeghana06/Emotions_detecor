import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
import os

def build_model():
    model = Sequential([
        Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(512, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.4),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(4, activation='softmax') 
    ])
    return model


MODEL_PATH = 'best_model.h5' 

if not os.path.exists(MODEL_PATH):
    print(f"Error: {MODEL_PATH} not found. Did you download it from Kaggle?")
    exit()

model = build_model()

try:
    
    model.load_weights(MODEL_PATH)
    print("Model weights loaded successfully!")
except Exception as e:
    print(f"Error loading weights: {e}")
    print("Trying load_model with compile=False...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)


face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam. Check Mac System Settings > Privacy & Security > Camera.")
    exit()

print("Starting Camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)   
        roi = np.expand_dims(roi, axis=-1)  

        preds = model.predict(roi, verbose=0)[0]
        label = emotion_labels[np.argmax(preds)]
        confidence = np.max(preds) * 100

        text = f"{label} ({confidence:.1f}%)"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()