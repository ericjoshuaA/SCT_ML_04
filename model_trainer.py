import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os

# Ensure the data directory exists
os.makedirs('models', exist_ok=True)

# Load data
data = pd.read_csv('data/gesture_data.csv', header=None)
X = data.iloc[:, :-1].values  # Landmark features
y = data.iloc[:, -1].values   # Gesture labels

num_classes = len(np.unique(y))
y_cat = to_categorical(y, num_classes)

# Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_cat, epochs=30, batch_size=32, validation_split=0.2)

# Save model and labels
model.save('models/gesture_model.h5')
with open('models/gesture_labels.txt', 'w') as f:
    for gesture in ['Fist', 'Palm', 'Peace']:
        f.write(f"{gesture}\n")
print("Model and labels saved.")