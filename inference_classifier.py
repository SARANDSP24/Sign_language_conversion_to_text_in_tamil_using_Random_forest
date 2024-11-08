import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image


# Load the Random Forest Classifier model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Tamil labels
labels_dict = {0:'அ',1:'ஆ',2:'இ',3:'ஈ',4:'உ',5:'ஊ',6:'எ',7:'ஏ',8:'ஐ',9:'ஒ',10:'ஓ',11:'ஔ',12:'ஃ',13:'சுழியம்',14:'ஒன்று',15:'இரண்டு',16:'மூன்று',17:'நான்கு',18:'ஐந்து',19:'ஆறு',20:'ஏழு',21:'எட்டு',22:'ஒன்பது',23:'பத்து',24:'வணக்கம்',25:'காலை',26:'பிற்பகல்',27:'இரவு',28:'சந்தோஷமாக',29:'வருத்தம்',30:'ஆண்',31:'பெண்',32:'அப்பா',33:'அம்மா',34:'சகோதரன்',35:'சகோதரி',36:'கணவன்',37:'மனைவி',38:'மகன்',39:'மகள்',40:'படிப்பு',41:'நான்',42:'என்னுடையது',43:'நீ',44:'இல்லை'}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from the camera.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Pad the feature vector to have 100 features
            while len(data_aux) < 100:
                data_aux.append(0.0)

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Display Tamil label with the specified font
            font_path = 'C:/Users/Saran/Downloads/sign-language-detector-python-master/catamaran/Catamaran-Bold.ttf'  # Replace with the actual path
            font_size = 30
            font = ImageFont.truetype(font_path, font_size)
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            draw.text((x1, y1 - 10), predicted_character, font=font, fill=(0, 0, 0))
            frame = np.array(img_pil)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
