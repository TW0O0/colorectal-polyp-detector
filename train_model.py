import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuration
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 30
MODEL_PATH = os.path.join('model', 'saved_models')
os.makedirs(MODEL_PATH, exist_ok=True)

def create_model(num_classes=3):
    """
    Create an EfficientNetB3 model with transfer learning
    Args:
        num_classes: Number of output classes (3 for normal, benign, cancerous)
    Returns:
        Compiled Keras model
    """
    # Create base model from EfficientNetB3
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add new classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Construct the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def prepare_data_generators(data_dir):
    """
    Create data generators for training, validation, and testing
    Args:
        data_dir: Path to dataset directory with train/val/test subdirectories
    Returns:
        Training, validation, and test data generators
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and testing
    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Set up the generators
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    valid_generator = valid_datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, valid_generator, test_generator

def fine_tune_model(model, train_generator, valid_generator, epochs=EPOCHS):
    """
    Two-stage training: train the head first, then fine-tune some of the base model
    Args:
        model: Initial model with frozen base layers
        train_generator: Training data generator
        valid_generator: Validation data generator
        epochs: Number of epochs for each stage
    Returns:
        Model and training history
    """
    # Phase 1: Train only the top layers
    print("Phase 1: Training the top layers...")
    history1 = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // BATCH_SIZE,
        epochs=epochs // 2,
        verbose=1
    )
    
    # Phase 2: Unfreeze the last 20 layers of the base model and train with lower learning rate
    print("Phase 2: Fine-tuning the last 20 layers...")
    for layer in model.layers[-20:]:
        layer.trainable = True
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    history2 = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // BATCH_SIZE,
        epochs=epochs // 2,
        verbose=1
    )
    
    # Combine histories
    combined_history = {}
    for k in history1.history.keys():
        combined_history[k] = history1.history[k] + history2.history[k]
    
    return model, combined_history

def evaluate_model(model, test_generator):
    """
    Evaluate the model on test data and generate performance metrics
    Args:
        model: Trained model
        test_generator: Test data generator
    """
    # Get predictions on test data
    y_pred_prob = model.predict(test_generator)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_generator.classes
    
    # Generate classification report
    class_names = list(test_generator.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:")
    print(report)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(MODEL_PATH, 'confusion_matrix.png'))
    
    # Save classification report to file
    with open(os.path.join(MODEL_PATH, 'classification_report.txt'), 'w') as f:
        f.write(report)

def plot_training_history(history):
    """
    Plot training and validation metrics
    Args:
        history: Training history
    """
    # Accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, 'training_history.png'))

def main():
    """Main function to train and evaluate the model"""
    # Check for data directory
    data_dir = 'data/colorectal_dataset'
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found.")
        print("Please download and organize colorectal polyp datasets before running this script.")
        return
    
    # Prepare data generators
    print("Preparing data...")
    train_generator, valid_generator, test_generator = prepare_data_generators(data_dir)
    
    # Create and train model
    print("Creating model...")
    model = create_model(num_classes=len(train_generator.class_indices))
    print("Training model...")
    model, history = fine_tune_model(model, train_generator, valid_generator)
    
    # Save model
    print("Saving model...")
    model.save(os.path.join(MODEL_PATH, 'efficientnet_colorectal_final.h5'))
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, test_generator)
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)
    
    print(f"Model training complete. Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
