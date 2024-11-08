# import os
# import cv2

# DATA_DIR = './data'
# if not os.path.exists(DATA_DIR):
#     os.makedirs(DATA_DIR)

# number_of_classes = 45
# dataset_size = 300

# cap = cv2.VideoCapture(0)
# for j in range(number_of_classes):
#     if not os.path.exists(os.path.join(DATA_DIR, str(j))):
#         os.makedirs(os.path.join(DATA_DIR, str(j)))

#     print('Collecting data for class {}'.format(j))

#     skip_class = False  # Flag to skip collecting data for this class
#     done = False
#     while True:
#         ret, frame = cap.read()
#         cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
#                     cv2.LINE_AA)
#         cv2.imshow('frame', frame)
#         key = cv2.waitKey(25)
#         if key == ord('q'):
#             break
#         elif key == ord('e'):
#             done = True
#             break
#         elif key == ord('s'):  # Press 's' to skip collecting data for this class
#             skip_class = True
#             break

#     if skip_class:
#         continue  # Skip collecting data for this class
#     if done:
#         break

#     counter = 0
#     while counter < dataset_size:
#         ret, frame = cap.read()
#         cv2.imshow('frame', frame)
#         cv2.waitKey(25)
#         cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

#         counter += 1

# cap.release()
# cv2.destroyAllWindows()




import cv2
import os
import time
import numpy as np
import Mediapipe as mp

# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Path where the videos will be stored
output_dir = 'videos'
create_directory(output_dir)

# Path where the landmark data will be stored
landmark_dir = 'landmarks'
create_directory(landmark_dir)

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Number of videos and video length (in seconds)
num_videos = int(input("Enter the number of videos to capture: "))
video_length = int(input("Enter the length of each video (in seconds): "))

# Frame rate (adjustable based on your needs)
fps = 20.0
frame_width = 640
frame_height = 480

# Initialize video capture
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

# Array to store video file paths
video_files = []

# Create a window to display capture status
status_window_name = "Capture Status"
cv2.namedWindow(status_window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(status_window_name, 400, 100)

def update_status_window(status_text):
    """Display the status in the status window."""
    status_img = np.zeros((100, 400, 3), np.uint8)
    cv2.putText(status_img, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(status_window_name, status_img)

for i in range(num_videos):
    # Video file name
    output_file = os.path.join(output_dir, f'output_{i+1}.mp4')
    video_files.append(output_file)
    
    # Landmark file name
    landmark_file = os.path.join(landmark_dir, f'landmarks_{i+1}.npy')

    update_status_window(f"Ready to capture video {i+1}/{num_videos}. Press 's' to start, 'k' to skip, or 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if ret:
            # Show the video frame in a window
            cv2.imshow('Video Capture', frame)

            # Wait for the user to press 's' to start/continue capturing, 'k' to skip, or 'q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # Start capturing
                update_status_window(f"Capturing video {i+1}/{num_videos}...")
                break
            elif key == ord('k'):  # Skip this video
                update_status_window(f"Skipped video {i+1}.")
                time.sleep(1)  # Short delay to show the skip status
                break
            elif key == ord('q'):  # Quit the process
                update_status_window("Quitting...")
                time.sleep(1)  # Short delay to show the quit status
                cap.release()
                cv2.destroyAllWindows()
                exit()

    # If the user skipped the video, continue to the next iteration
    if key == ord('k'):
        continue

    print(f"Capturing video {i+1}/{num_videos}...")

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    # Calculate the number of frames to capture based on the video length and FPS
    num_frames = int(fps * video_length)

    # Initialize MediaPipe holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # List to store landmarks data
        landmarks_data = []

        for _ in range(num_frames):
            ret, frame = cap.read()

            if ret:
                # Convert the BGR image to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame with MediaPipe
                results = holistic.process(rgb_frame)

                # Draw landmarks on the frame for visualization
                mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # Save the landmarks data
                if results.pose_landmarks:
                    pose_landmarks = results.pose_landmarks.landmark
                    face_landmarks = results.face_landmarks.landmark if results.face_landmarks else []
                    left_hand_landmarks = results.left_hand_landmarks.landmark if results.left_hand_landmarks else []
                    right_hand_landmarks = results.right_hand_landmarks.landmark if results.right_hand_landmarks else []
                    landmarks_data.append({
                        'pose': [(lm.x, lm.y, lm.z) for lm in pose_landmarks],
                        'face': [(lm.x, lm.y, lm.z) for lm in face_landmarks],
                        'left_hand': [(lm.x, lm.y, lm.z) for lm in left_hand_landmarks],
                        'right_hand': [(lm.x, lm.y, lm.z) for lm in right_hand_landmarks]
                    })

                # Write the frame to the video file
                out.write(frame)
                cv2.imshow('Video Capture', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    update_status_window("Quitting...")
                    time.sleep(1)
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()
                    exit()
            else:
                break

        # Save the landmarks data as a NumPy file
        np.save(landmark_file, landmarks_data)

    # Release the video writer for the current video
    out.release()

    update_status_window(f"Video {i+1} and landmarks saved.")
    print(f"Video {i+1} saved as {output_file}")
    print(f"Landmarks for video {i+1} saved as {landmark_file}")

    # Optional: Delay between videos
    time.sleep(2)  # 2-second delay between videos

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Print the list of video files captured and their corresponding landmarks
print("Captured videos and landmarks:")
for video_file in video_files:
    print(video_file)
    print(f"Landmarks: {os.path.join(landmark_dir, os.path.basename(video_file).replace('.mp4', '.npy'))}")
