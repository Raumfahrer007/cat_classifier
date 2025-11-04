import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np

def create_data_generators(target_size=(224, 224), batch_size=32, validation_split=0.15):
    """
    Create data generators with handling for class imbalance
    """
    # Calculate class weights to handle imbalance
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Training generator with heavy augmentation for minority classes
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        rotation_range=25,  # More augmentation for minority
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,  # Extra for minority
        zoom_range=0.3,
        brightness_range=[0.7, 1.3],
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    return train_datagen, datagen

def calculate_class_weights(data_path, cat_classes):
    """
    Calculate class weights to handle imbalance during training
    """
    class_counts = {}
    
    for cat_name, cat_key in cat_classes.items():
        class_path = os.path.join(data_path, cat_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[cat_key] = len(image_files)
    
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    class_weights = {}
    for cat_name, cat_key in cat_classes.items():
        # Higher weight for underrepresented classes
        class_weights[cat_key] = total_samples / (num_classes * class_counts[cat_key])
    
    print("Class weights for handling imbalance:", class_weights)
    return class_weights