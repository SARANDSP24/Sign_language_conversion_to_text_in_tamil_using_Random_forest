Project Overview


This project aims to develop a machine learning solution for translating Tamil sign language gestures into readable Tamil text, bridging communication gaps for the deaf and hard-of-hearing community. The application uses computer vision and a Random Forest classifier to recognize specific gestures from video frames and convert them into textual form.

Objectives


Recognize Tamil Sign Language: Detect and classify gestures specific to Tamil sign language.
Translate to Text: Convert recognized gestures into readable Tamil text.


Improve Accessibility: Create a communication tool that assists in bridging language barriers for the deaf and hard-of-hearing community.


Features


Real-time Gesture Recognition: Leverages computer vision for real-time video processing and gesture recognition.


Random Forest Classifier: Utilizes a Random Forest model to classify each gesture based on extracted features from video frames.


Tamil Text Output: Converts recognized gestures into Tamil text, ensuring an accessible output for native speakers.


Project Workflow


Data Collection: Videos of Tamil sign language gestures are collected and processed.


Frame Extraction: Each video is segmented into frames to capture individual gestures.


Feature Extraction: Key features are extracted from each frame, which serve as inputs to the Random Forest model.


Model Training: The Random Forest classifier is trained on labeled gesture data.


Inference and Translation: The model predicts the gestures in real-time and converts them into Tamil text.


The Random Forest model is trained on features extracted from video frames. You may adjust the hyperparameters and retrain the model for improved accuracy by modifying the training script provided in the train_model.py file.
