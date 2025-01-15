# import cv2
# import os

# def get_face_detector():
#     # Load a pre-trained Haar Cascade classifier for face detection
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     return face_cascade

# def load_video_and_detect_faces(video_path, output_folder):
#     # Initialize the face detector
#     face_cascade = get_face_detector()
    
#     # Capture the video
#     cap = cv2.VideoCapture(video_path)
    
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return
    
#     # Get total number of frames in the video
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"Total frames in video: {total_frames}")
    
#     # Calculate number of frames to extract (30% of total frames)
#     frames_to_extract = int(total_frames * 0.3)
    
#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)

#     frame_count = 0
#     saved_frame_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break  # Exit if there are no more frames
        
#         # Process only every nth frame based on the calculated ratio
#         if frame_count % (total_frames // frames_to_extract) == 0:
#             # Convert the frame to grayscale (face detection works better in grayscale)
#             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
#             # Detect faces in the frame
#             faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
            
#             # Draw rectangles around detected faces (optional)
#             for (x, y, w, h) in faces:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
#             # Save the processed frame to the output folder
#             output_frame_path = os.path.join(output_folder, f'frame_{saved_frame_count}.jpg')
#             cv2.imwrite(output_frame_path, frame)
#             saved_frame_count += 1
            
#             # Display the frame with detected faces (optional)
#             cv2.imshow('Video', frame)

#         frame_count += 1
        
#         # Press 'q' to exit the video window early
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the video capture object and close all OpenCV windows
#     cap.release()
#     cv2.destroyAllWindows()

# # Example usage
# video_file_path = 'trial_lie_001.mp4'  # Replace with your video file path
# output_folder_path = 'outputs'         # Folder to save extracted frames
# load_video_and_detect_faces(video_file_path, output_folder_path)

# Step 1 : this works

# import cv2
# import os

# def get_face_detector():
#     # Load a pre-trained Haar Cascade classifier for face detection
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     return face_cascade

# def load_video_and_detect_faces(video_path, output_folder):
#     # Initialize the face detector
#     face_cascade = get_face_detector()
    
#     # Capture the video
#     cap = cv2.VideoCapture(video_path)
    
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return
    
#     # Get total number of frames in the video
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"Total frames in video: {total_frames}")
    
#     # Calculate how many frames to skip to get approximately 30%
#     skip_frames = int(total_frames * 0.7)  # Skip 70% of frames

#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)

#     frame_count = 0
#     saved_frame_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break  # Exit if there are no more frames
        
#         # Save only every nth frame based on calculated skip
#         if frame_count % (skip_frames // (total_frames // 30)) == 0:
#             # Convert the frame to grayscale (face detection works better in grayscale)
#             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
#             # Detect faces in the frame
#             faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
            
#             # Draw rectangles around detected faces (optional)
#             for (x, y, w, h) in faces:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
#             # Save the processed frame to the output folder
#             output_frame_path = os.path.join(output_folder, f'frame_{saved_frame_count}.jpg')
#             cv2.imwrite(output_frame_path, frame)
#             saved_frame_count += 1
            
#             # Display the frame with detected faces (optional)
#             cv2.imshow('Video', frame)

#         frame_count += 1
        
#         # Press 'q' to exit the video window early
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the video capture object and close all OpenCV windows
#     cap.release()
#     cv2.destroyAllWindows()

# # Example usage
# video_file_path = 'trial_lie_001.mp4'  # Replace with your video file path
# output_folder_path = 'outputs'         # Folder to save extracted frames
# load_video_and_detect_faces(video_file_path, output_folder_path)


# This is good. saves the crops to a folder naming it "sad", "happy"
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

def get_face_detector():
    # Load a pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def load_emotion_model(model_path):
    # Load the pre-trained emotion detection model
    model = load_model(model_path)
    return model

def preprocess_face(face_image):
    # Resize and normalize the face image for emotion detection
    face_image = cv2.resize(face_image, (48, 48))  # Assuming model expects 48x48 input size
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    face_image = face_image / 255.0  # Normalize to [0, 1]
    face_image = np.expand_dims(face_image, axis=-1)  # Add channel dimension
    face_image = np.expand_dims(face_image, axis=0)  # Add batch dimension
    return face_image

def load_video_and_detect_faces(video_path, output_folder, emotion_model):
    # Initialize the face detector
    face_cascade = get_face_detector()
    
    # Capture the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    
    skip_frames = int(total_frames * 0.7)  # Skip 70% of frames

    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % (skip_frames // (total_frames // 30)) == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Crop the detected face region
                face_crop = frame[y:y+h, x:x+w]
                
                # Preprocess the cropped face for emotion detection
                processed_face = preprocess_face(face_crop)
                
                # Predict emotion using the loaded model
                emotion_prediction = emotion_model.predict(processed_face)
                predicted_emotion = np.argmax(emotion_prediction)  # Get index of max probability
                
                # Map index to emotion label (adjust according to your model's labels)
                emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                emotion_text = emotion_labels[predicted_emotion]

                # Draw rectangle around detected face and put text with predicted emotion
                cv2.rectangle(face_crop, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(face_crop, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            output_frame_path = os.path.join(output_folder, f'frame_{saved_frame_count}_{emotion_text}.jpg')
            cv2.imwrite(output_frame_path, face_crop)
            saved_frame_count += 1
            
            # cv2.imshow('Video', frame)

        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_file_path = 'trial_lie_001.mp4'  # Replace with your video file path
output_folder_path = 'outputs'         # Folder to save extracted frames
emotion_model_path = 'model/face_model.h5'  # Replace with your model path

# Load the emotion detection model
emotion_model = load_emotion_model(emotion_model_path)

# Run the video processing function
load_video_and_detect_faces(video_file_path, output_folder_path, emotion_model)

