import os 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
 
 # Set the directory paths 
train_dir = '/kaggle/input/natural-images/natural_images' 
test_dir = '/kaggle/input/natural-images/natural_images' 
 
 # Define image dimensions and batch size 
img_width, img_height = 150, 150 
batch_size = 32 
 
# Preprocess and augment the training images 
train_datagen = ImageDataGenerator( 
    rescale=1./255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True) 
 
# Load and preprocess the test images 
test_datagen = ImageDataGenerator(rescale=1./255) 
 
 
# Generate batches of augmented data for training 
train_generator = train_datagen.flow_from_directory( 
    train_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode='binary') 
 
# Generate batches of augmented data for testing 
test_generator = test_datagen.flow_from_directory( 
    test_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode='binary') 
 
 
# Build the CNN model 
model = Sequential([ 
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)), 
    MaxPooling2D((2, 2)), 
    Conv2D(64, (3, 3), activation='relu'), 
    MaxPooling2D((2, 2)), 
    Conv2D(128, (3, 3), activation='relu'), 
    MaxPooling2D((2, 2)), 
    Flatten(), 
    Dense(512, activation='relu'), 
    Dense(1, activation='sigmoid') 
]) 
 
 
# Compile the model 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
 
# Train the model 
model.fit( 
    train_generator, 
    steps_per_epoch=train_generator.samples // batch_size, 
    epochs=1, 
    validation_data=test_generator, 
    validation_steps=test_generator.samples // batch_size 
) 
 
# Evaluate the model 
loss, accuracy = model.evaluate(test_generator) 
print("Test Accuracy: {:.2f}%".format(accuracy * 100)) 
 
labels = os.listdir(train_dir) 
num = [] 
for label in labels: 
    path = os.path.join(train_dir, label) 
    num.append(len(os.listdir(path))) 
 
# Plot the number of images in each class 
import matplotlib.pyplot as plt 
import plotly.graph_objects as go 
fig = go.Figure(data=[go.Bar( 
    x=labels,  
    y=num, 
    
    text=num, 
    textposition='auto', 
)]) 
fig.update_layout(title_text='NUMBER OF IMAGES CONTAINED IN EACH 
CLASS') 
fig.show()