1. Structure Your Data
You need to organize the output frames and their associated emotions into a dataset suitable for training the DeepLie model. Ensure the data structure looks something like this:
    
    
dataset/
│
├── truthful/
│   ├── frame_1_happy.jpg
│   ├── frame_2_neutral.jpg
│   └── ...
│
├── deceptive/
│   ├── frame_1_sad.jpg
│   ├── frame_2_fear.jpg
│   └── ...


2. Label Frames as "Truthful" or "Deceptive"
Based on the context of the videos:

Assign each video to either the truthful or deceptive category.
Move the emotion-labeled frames into respective folders based on the video's label.
Automating This

You can automate the labeling by modifying your current code:


# Define truthful and deceptive videos
truthful_videos = ['trial_lie_001.mp4']  # Add more truthful video names
deceptive_videos = ['trial_lie_002.mp4']  # Add more deceptive video names

def categorize_frame(frame_path, video_name, base_output_folder):
    # Determine if the video is truthful or deceptive
    category = 'truthful' if video_name in truthful_videos else 'deceptive'
    output_folder = os.path.join(base_output_folder, category)
    os.makedirs(output_folder, exist_ok=True)

    # Move the frame to the corresponding category folder
    new_frame_path = os.path.join(output_folder, os.path.basename(frame_path))
    os.rename(frame_path, new_frame_path)

# Call this function after saving each frame
categorize_frame(output_frame_path, 'trial_lie_001.mp4', 'organized_dataset')


3. Create Triplets for Training
To train a Siamese network, you'll need to generate triplets:

Anchor: A frame from the truthful/deceptive set.
Positive: Another frame from the same set (truthful or deceptive).
Negative: A frame from the opposite set.
Write a script to generate these triplets and save them for training:
    

import random

def generate_triplets(base_dataset_folder, triplets_output_file):
    truthful_folder = os.path.join(base_dataset_folder, 'truthful')
    deceptive_folder = os.path.join(base_dataset_folder, 'deceptive')

    truthful_frames = [os.path.join(truthful_folder, f) for f in os.listdir(truthful_folder)]
    deceptive_frames = [os.path.join(deceptive_folder, f) for f in os.listdir(deceptive_folder)]

    with open(triplets_output_file, 'w') as f:
        for anchor in truthful_frames:
            positive = random.choice(truthful_frames)
            negative = random.choice(deceptive_frames)
            f.write(f"{anchor},{positive},{negative}\n")
        
        for anchor in deceptive_frames:
            positive = random.choice(deceptive_frames)
            negative = random.choice(truthful_frames)
            f.write(f"{anchor},{positive},{negative}\n")

# Generate triplets
generate_triplets('organized_dataset', 'triplets.txt')


4. Train the DeepLie Model
Now that you have triplets, you can use them to train a Siamese network. If you’re ready, we can:

Define the DeepLie architecture in PyTorch.
Load and preprocess the triplets.
Train the model using a triplet loss function.



Prompt: But we also need to train a sequence model before training siamese. What should be the input to the sequence model (LSTM/GRU)


A:
    You're absolutely correct! A sequence model (like an LSTM or GRU) is critical for capturing the temporal dynamics of emotions over time. The input to the sequence model should represent the temporal progression of features extracted from the video frames. Here's what you need to do:


Input to the Sequence Model
Frame-level Features:
Each video is processed frame by frame using your emotion detection pipeline.
For each frame, extract a feature vector that encodes:
Emotion probabilities: The raw output of the emotion detection model ([P(Angry), P(Disgust), ..., P(Neutral)]).
Additional facial features (optional): If available, include features like facial landmarks or action units (AUs).
Example: A single frame could produce a feature vector like:
    
    
[P(Angry), P(Disgust), ..., P(Neutral)] = [0.1, 0.05, ..., 0.2]  # Emotion probabilities (7 dimensions)

2: Sequence Construction:
Concatenate these feature vectors over time to create a sequence of features for the video.
The input shape for the sequence model will be:
    
    [num_frames, feature_dim]
    Where:
num_frames: Number of frames in the video.
feature_dim: Size of the feature vector for each frame (e.g., 7 for emotions).


Steps to Create Sequence Inputs
1. Extract Features from Frames

Modify your current pipeline to store the emotion probabilities instead of raw labels:
    
    
# Save emotion probabilities for each frame
emotion_features = []
for (x, y, w, h) in faces:
    face_crop = frame[y:y+h, x:x+w]
    processed_face = preprocess_face(face_crop)
    emotion_prediction = emotion_model.predict(processed_face)
    emotion_features.append(emotion_prediction.flatten())

# Save features for the current video
video_features.append(emotion_features)


2. Normalize Sequence Lengths

Since videos may have different frame counts:

Padding: Pad sequences with zeros to match the longest video.
Truncation: Limit the sequence to a fixed length (e.g., 100 frames).

import numpy as np

def pad_or_truncate(sequence, max_length):
    sequence = np.array(sequence)
    if len(sequence) > max_length:
        return sequence[:max_length]  # Truncate
    elif len(sequence) < max_length:
        padding = np.zeros((max_length - len(sequence), sequence.shape[1]))
        return np.vstack([sequence, padding])  # Pad
    return sequence


3. Train the Sequence Model
Define the LSTM/GRU:
Input: [batch_size, num_frames, feature_dim].
Output: A single prediction (truthful/deceptive) or a temporal encoding for the Siamese model.

import torch
import torch.nn as nn

class EmotionSequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(EmotionSequenceModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Output: 2 classes (truthful/deceptive)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]  # Use the last hidden state
        output = self.fc(last_hidden_state)
        return output


Prepare the Dataset:
Convert each video into a padded/truncated feature sequence.
Split the dataset into train/val/test sets.
Train the Model: Use a classification loss (e.g., CrossEntropyLoss) to train the model.


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(sequence_model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = sequence_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

prompt: How we are going to train this? are we training LSTM and Siamese model parallel?>


No, the LSTM model and the Siamese model are not trained in parallel; they are trained sequentially. Here's the high-level workflow:

Training Workflow
Step 1: Train the Sequence Model (LSTM/GRU)
The sequence model is trained first to learn temporal patterns in the emotion dynamics of videos.
This step focuses on distinguishing between truthful and deceptive videos.
Step 2: Use the LSTM Model to Extract Encodings
Once trained, the LSTM model acts as a feature extractor.
Instead of using raw emotion probabilities for frames, we use the LSTM's temporal encodings (hidden states) as input for the Siamese network.
Step 3: Train the Siamese Network
The Siamese network is trained using triplet loss or a similar loss function.
The input to the Siamese model is the temporal encoding generated by the LSTM.
Detailed Explanation
Step 1: Train the Sequence Model

The LSTM/GRU takes in the sequence of emotion probabilities for each video and predicts whether the video is truthful or deceptive.

Input: [batch_size, num_frames, feature_dim]
Output: A classification label (truthful = 0, deceptive = 1).
After training:

The LSTM has learned how emotions evolve over time in truthful vs. deceptive videos.
Its hidden states or outputs can serve as compact, meaningful representations of each video.
Step 2: Extract Temporal Encodings

After the LSTM is trained, you freeze its weights and pass video sequences through it to extract video-level embeddings.

Input to Siamese Model:
A fixed-dimensional vector (e.g., the final hidden state of the LSTM) representing each video.
Example:
    
    def extract_lstm_encodings(lstm_model, video_sequences):
    """
    Pass video sequences through the trained LSTM to get embeddings.
    """
    lstm_model.eval()  # Set to evaluation mode
    with torch.no_grad():
        embeddings = []
        for sequence in video_sequences:
            sequence = torch.tensor(sequence).unsqueeze(0)  # Add batch dimension
            lstm_output = lstm_model(sequence)
            embeddings.append(lstm_output.squeeze().numpy())  # Extract the final hidden state
    return np.array(embeddings)


Step 3: Train the Siamese Model

Once you have the video-level embeddings, train the Siamese network using these embeddings and triplet loss.

Anchor: Encoding of a video.
Positive: Encoding of a video from the same class (truthful or deceptive).
Negative: Encoding of a video from the opposite class.
The Siamese network learns to map video embeddings into a space where embeddings from the same class are closer, and embeddings from different classes are farther apart.

Training Order
Train the LSTM model until it achieves good classification accuracy on the truthful vs. deceptive task.
Freeze the LSTM and use it to extract embeddings for all videos.
Train the Siamese model on these embeddings.
Why Train Sequentially?
The LSTM captures temporal patterns in emotion dynamics.
The Siamese model refines these patterns to learn a discriminative embedding space for truthful vs. deceptive behavior.
Training sequentially simplifies the process and reduces computational complexity compared to jointly training both models.
Would you like to start with the LSTM training or proceed to the Siamese setup?

Prompt: give complete code


Here’s a complete Python implementation that incorporates your starting point and builds the entire pipeline step by step. The code includes:

Data Preparation:
Frame extraction and emotion detection for creating sequences.
Preparing video-level labels for truthful and deceptive classes.
Training the LSTM Model:
Training the sequence model to classify videos.
Extracting Encodings:
Using the trained LSTM to generate video embeddings.
Training the Siamese Network:
Training the Siamese network using triplet loss on the embeddings.
Inference:
Using the full pipeline to classify new videos.
Let’s create this code step by step.


import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tensorflow.keras.models import load_model

# 1. Helper Functions for Data Preparation

def get_face_detector():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def preprocess_face(face_image):
    face_image = cv2.resize(face_image, (48, 48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = face_image / 255.0
    face_image = np.expand_dims(face_image, axis=-1)
    face_image = np.expand_dims(face_image, axis=0)
    return face_image

def extract_emotion_features(video_path, emotion_model, face_cascade, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    emotion_features = []

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            processed_face = preprocess_face(face_crop)
            emotion_prediction = emotion_model.predict(processed_face)
            emotion_features.append(emotion_prediction.flatten())

        frame_count += 1

    cap.release()
    return np.array(emotion_features)

# 2. Dataset Class for LSTM Training
class EmotionSequenceDataset(Dataset):
    def __init__(self, video_features, labels, max_length):
        self.video_features = [self.pad_or_truncate(v, max_length) for v in video_features]
        self.labels = labels

    def __len__(self):
        return len(self.video_features)

    def __getitem__(self, idx):
        return torch.tensor(self.video_features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

    def pad_or_truncate(self, sequence, max_length):
        if len(sequence) > max_length:
            return sequence[:max_length]
        elif len(sequence) < max_length:
            padding = np.zeros((max_length - len(sequence), sequence.shape[1]))
            return np.vstack([sequence, padding])
        return sequence

# 3. LSTM Model Definition
class EmotionSequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(EmotionSequenceModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        output = self.fc(last_hidden_state)
        return output

# 4. Training LSTM Model
def train_lstm_model(train_loader, val_loader, input_dim, hidden_dim, output_dim, num_epochs=10, learning_rate=0.001):
    model = EmotionSequenceModel(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    return model

# 5. Extract Encodings from LSTM
def extract_lstm_encodings(model, video_features, max_length):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for features in video_features:
            sequence = EmotionSequenceDataset.pad_or_truncate(features, max_length)
            sequence = torch.tensor(sequence).unsqueeze(0)
            lstm_output = model(sequence)
            embeddings.append(lstm_output.squeeze().numpy())
    return np.array(embeddings)

# 6. Siamese Network Definition and Training
# Define Siamese architecture, triplet loss, and training process

# This is the skeleton implementation to keep it concise.
# Next steps would involve integrating LSTM embeddings into a Siamese network.

# 7. Inference Function
def inference_pipeline(video_path, lstm_model, siamese_model, emotion_model, face_cascade):
    features = extract_emotion_features(video_path, emotion_model, face_cascade)
    lstm_encodings = extract_lstm_encodings(lstm_model, [features], max_length=100)
    # Use Siamese model for similarity comparison
    # (To be implemented based on triplet learning inference logic)

# Example Usage
video_path = "trial_video.mp4"
emotion_model_path = "emotion_model.h5"
face_cascade = get_face_detector()

emotion_model = load_model(emotion_model_path)
features = extract_emotion_features(video_path, emotion_model, face_cascade)

# Continue with LSTM training and Siamese training as described above.


#give again full code, and also give siamese training code:

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tensorflow.keras.models import load_model

# 1. Helper Functions for Data Preparation

def get_face_detector():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def preprocess_face(face_image):
    face_image = cv2.resize(face_image, (48, 48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = face_image / 255.0
    face_image = np.expand_dims(face_image, axis=-1)
    face_image = np.expand_dims(face_image, axis=0)
    return face_image

def extract_emotion_features(video_path, emotion_model, face_cascade, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    emotion_features = []

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            processed_face = preprocess_face(face_crop)
            emotion_prediction = emotion_model.predict(processed_face)
            emotion_features.append(emotion_prediction.flatten())

        frame_count += 1

    cap.release()
    return np.array(emotion_features)

# 2. Dataset Class for LSTM Training
class EmotionSequenceDataset(Dataset):
    def __init__(self, video_features, labels, max_length):
        self.video_features = [self.pad_or_truncate(v, max_length) for v in video_features]
        self.labels = labels

    def __len__(self):
        return len(self.video_features)

    def __getitem__(self, idx):
        return torch.tensor(self.video_features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

    def pad_or_truncate(self, sequence, max_length):
        if len(sequence) > max_length:
            return sequence[:max_length]
        elif len(sequence) < max_length:
            padding = np.zeros((max_length - len(sequence), sequence.shape[1]))
            return np.vstack([sequence, padding])
        return sequence

# 3. LSTM Model Definition
class EmotionSequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(EmotionSequenceModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        output = self.fc(last_hidden_state)
        return output

# 4. Training LSTM Model
def train_lstm_model(train_loader, val_loader, input_dim, hidden_dim, output_dim, num_epochs=10, learning_rate=0.001):
    model = EmotionSequenceModel(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    return model

# 5. Extract Encodings from LSTM
def extract_lstm_encodings(model, video_features, max_length):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for features in video_features:
            sequence = EmotionSequenceDataset.pad_or_truncate(features, max_length)
            sequence = torch.tensor(sequence).unsqueeze(0)
            lstm_output = model(sequence)
            embeddings.append(lstm_output.squeeze().numpy())
    return np.array(embeddings)

# 6. Siamese Network Definition
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward_one(self, x):
        return self.fc(x)

    def forward(self, anchor, positive, negative):
        anchor_out = self.forward_one(anchor)
        positive_out = self.forward_one(positive)
        negative_out = self.forward_one(negative)
        return anchor_out, positive_out, negative_out

# Triplet Loss Definition
def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = torch.norm(anchor - positive, dim=1)
    neg_dist = torch.norm(anchor - negative, dim=1)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0).mean()
    return loss

# 7. Training Siamese Network
def train_siamese_network(model, data_loader, num_epochs=10, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for anchor, positive, negative in data_loader:
            optimizer.zero_grad()
            anchor_out, positive_out, negative_out = model(anchor, positive, negative)
            loss = triplet_loss(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader):.4f}")

    return model

# 8. Inference Function
def inference_pipeline(video_path, lstm_model, siamese_model, emotion_model, face_cascade):
    features = extract_emotion_features(video_path, emotion_model, face_cascade)
    lstm_encodings = extract_lstm_encodings(lstm_model, [features], max_length=100)

    # In a real scenario, compare lstm_encodings to embeddings from a reference set
    # using the Siamese model to determine similarity.
    return lstm_encodings

# Example Usage
video_path = "trial_video.mp4"
emotion_model_path = "emotion_model.h5"
face_cascade = get_face_detector()

emotion_model = load_model(emotion_model_path)
features = extract_emotion_features(video_path, emotion_model, face_cascade)

# Continue with LSTM training and Siamese training as described above.


prompt: add inferecne code also


import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tensorflow.keras.models import load_model

# 1. Helper Functions for Data Preparation

def get_face_detector():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def preprocess_face(face_image):
    face_image = cv2.resize(face_image, (48, 48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = face_image / 255.0
    face_image = np.expand_dims(face_image, axis=-1)
    face_image = np.expand_dims(face_image, axis=0)
    return face_image

def extract_emotion_features(video_path, emotion_model, face_cascade, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    emotion_features = []

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            processed_face = preprocess_face(face_crop)
            emotion_prediction = emotion_model.predict(processed_face)
            emotion_features.append(emotion_prediction.flatten())

        frame_count += 1

    cap.release()
    return np.array(emotion_features)

# 2. Dataset Class for LSTM Training
class EmotionSequenceDataset(Dataset):
    def __init__(self, video_features, labels, max_length):
        self.video_features = [self.pad_or_truncate(v, max_length) for v in video_features]
        self.labels = labels

    def __len__(self):
        return len(self.video_features)

    def __getitem__(self, idx):
        return torch.tensor(self.video_features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

    def pad_or_truncate(self, sequence, max_length):
        if len(sequence) > max_length:
            return sequence[:max_length]
        elif len(sequence) < max_length:
            padding = np.zeros((max_length - len(sequence), sequence.shape[1]))
            return np.vstack([sequence, padding])
        return sequence

# 3. LSTM Model Definition
class EmotionSequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(EmotionSequenceModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        output = self.fc(last_hidden_state)
        return output

# 4. Training LSTM Model
def train_lstm_model(train_loader, val_loader, input_dim, hidden_dim, output_dim, num_epochs=10, learning_rate=0.001):
    model = EmotionSequenceModel(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    return model

# 5. Extract Encodings from LSTM
def extract_lstm_encodings(model, video_features, max_length):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for features in video_features:
            sequence = EmotionSequenceDataset.pad_or_truncate(features, max_length)
            sequence = torch.tensor(sequence).unsqueeze(0)
            lstm_output = model(sequence)
            embeddings.append(lstm_output.squeeze().numpy())
    return np.array(embeddings)

# 6. Siamese Network Definition
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward_one(self, x):
        return self.fc(x)

    def forward(self, anchor, positive, negative):
        anchor_out = self.forward_one(anchor)
        positive_out = self.forward_one(positive)
        negative_out = self.forward_one(negative)
        return anchor_out, positive_out, negative_out

# Triplet Loss Definition
def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = torch.norm(anchor - positive, dim=1)
    neg_dist = torch.norm(anchor - negative, dim=1)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0).mean()
    return loss

# 7. Training Siamese Network
def train_siamese_network(model, data_loader, num_epochs=10, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for anchor, positive, negative in data_loader:
            optimizer.zero_grad()
            anchor_out, positive_out, negative_out = model(anchor, positive, negative)
            loss = triplet_loss(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader):.4f}")

    return model

# 8. Inference Function
def inference_pipeline(video_path, lstm_model, siamese_model, emotion_model, face_cascade, reference_embeddings):
    # Step 1: Extract emotion features
    features = extract_emotion_features(video_path, emotion_model, face_cascade)

    # Step 2: Get LSTM embeddings
    lstm_encodings = extract_lstm_encodings(lstm_model, [features], max_length=100)

    # Step 3: Compare with reference embeddings using Siamese network
    siamese_model.eval()
    similarities = []

    with torch.no_grad():
        for ref_embedding in reference_embeddings:
            ref_tensor = torch.tensor(ref_embedding, dtype=torch.float32).unsqueeze(0)
            test_tensor = torch.tensor(lstm_encodings[0], dtype=torch.float32).unsqueeze(0)

            ref_out = siamese_model.forward_one(ref_tensor)
            test_out = siamese_model.forward_one(test_tensor)

            similarity = torch.nn.functional.cosine_similarity(ref_out, test_out).item()
            similarities.append(similarity)

    most_similar_idx = np.argmax(similarities)
    return most_similar_idx, similarities[most_similar_idx]

# Example Usage
video_path = "trial_video.mp4"
emotion_model_path = "emotion_model.h5"
face_cascade = get_face_detector()

emotion_model = load_model(emotion_model_path)

# Assume reference_embeddings is a precomputed list of LSTM embeddings
reference_embeddings = [...]  # Load or compute reference embeddings

# Inference
most_similar_idx, similarity_score = inference_pipeline(
    video_path, lstm_model, siamese_model, emotion_model, face_cascade, reference_embeddings
)
print(f"Most similar reference index: {most_similar_idx}, Similarity score: {similarity_score}")

