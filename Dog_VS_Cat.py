'''Loading the Data'''

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Paths to directories containing cat and dog images
dog_dir = "C:/Users/kkkkk/Desktop/Projects/Dog VS Cat/dataset/Dog/first100"
cat_dir = "C:/Users/kkkkk/Desktop/Projects/Dog VS Cat/dataset/Cat/first100"

# Image size for resizing (e.g., 128x128)
IMG_SIZE = 128

def load_images(directory, label):
    images = []
    labels = []
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        try:
            # Load image and resize
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
    return images, labels

# Load dog and cat images
dog_images, dog_labels = load_images(dog_dir, 1)  # Label for dog is 1
cat_images, cat_labels = load_images(cat_dir, 0)  # Label for cat is 0

# Combine and convert to numpy arrays
X = np.array(dog_images + cat_images)
y = np.array(dog_labels + cat_labels)

# Normalize the images (pixel values between 0 and 1)
X = X / 255.0

'''Splitting the Data'''

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets (e.g., 10% validation)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")

'''Data Augmentation '''

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define a data generator for augmentation
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Apply it to the training data
train_datagen.fit(X_train)

# Create a simple generator for validation (no augmentation)
val_datagen = ImageDataGenerator()

'''Building the CNN Model'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the model
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten and Dense layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))  # Dropout to prevent overfitting
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

'''Model Training'''

# Train the model
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=20,
    verbose=1
)

'''Model Evaluation'''

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

# Predictions on test set
predictions = model.predict(X_test)

'''Visualizing Training Results'''

# import matplotlib.pyplot as plt

# # Plot training & validation accuracy and loss
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(len(acc))

# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')

# plt.show()

'''Testing on New Images'''

import numpy as np
from tensorflow.keras.preprocessing import image

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    
    prediction = model.predict(img_array)
    
    if prediction[0][0] > 0.5:
        print(f"Prediction: Dog (Confidence: {prediction[0][0]:.2f})")
    else:
        print(f"Prediction: Cat (Confidence: {1 - prediction[0][0]:.2f})")

# Test with a new image
predict_image("C:/Users/kkkkk/Desktop/Projects/Dog VS Cat/dataset/Dog/6.jpg")

'''Saving the Trained Model'''

# Save the model
model.save('dog_cat_classifier_model.h5')