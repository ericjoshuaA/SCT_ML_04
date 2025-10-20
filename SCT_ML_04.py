import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- 1. Generate Synthetic Data ---
def generate_synthetic_data(num_samples_per_gesture=100):
    """
    Generates synthetic hand landmark data for three gestures:
    'fist', 'open_hand', and 'peace_sign'.
    """
    gestures = ['fist', 'open_hand', 'peace_sign']
    landmarks = []
    labels = []
    
    # A hand has 21 landmarks, each with x, y, z coordinates.
    # So, 21 * 3 = 63 features.
    num_features = 63 

    # Generate data for each gesture
    for gesture in gestures:
        for _ in range(num_samples_per_gesture):
            # Generate a base set of random landmarks
            base_landmarks = np.random.rand(num_features).tolist()

            # Add a bit of 'noise' to simulate slight variations in real data
            noise = np.random.uniform(-0.05, 0.05, num_features)
            noisy_landmarks = (np.array(base_landmarks) + noise).tolist()
            
            # This is where we'd add 'rules' to make the data more realistic,
            # for example, enforcing certain joint positions for a 'fist'.
            # For this simple example, we'll just generate random data.
            landmarks.append(noisy_landmarks)
            labels.append(gesture)
            
    return np.array(landmarks), np.array(labels), gestures

# Generate our dataset
X, y_labels, gesture_labels = generate_synthetic_data(num_samples_per_gesture=500)

# --- 2. Preprocess Data for Model Training ---
le = LabelEncoder()
y = le.fit_transform(y_labels)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Build the Neural Network Model ---
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(gesture_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 4. Train the Model ---
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=32, verbose=2)

# --- 5. Save the Trained Model and Label Encoder ---
# Create the 'models' directory if it doesn't exist
import os
if not os.path.exists('models'):
    os.makedirs('models')

model.save('models/gesture_model.h5')

# We'll save the gesture labels directly since we don't have a label encoder object.
# This makes it easier to use in the real-time script.
with open('models/gesture_labels.txt', 'w') as f:
    for label in gesture_labels:
        f.write(label + '\n')

print("Model training complete and saved.")