import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import seaborn as sns
import os

# Constants
IMG_SIZE = (224, 224)  # InceptionV3 default is (299, 299) but works with 224
BATCH_SIZE = 32
NUM_CLASSES = 4  # Healthy, Tea leaf blight, Tea red leaf spot, Tea red scab
EPOCHS = 30
DATA_DIR = "D:/Browns/Tea Leaves Project/Tea leaf dataset"

# Enhanced Data Augmentation with proper InceptionV3 preprocessing
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Proper InceptionV3 scaling (-1 to 1)
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True,
    vertical_flip=False,  # Usually not natural for leaves
    fill_mode='reflect',
    channel_shift_range=5.0
)

# Train and validation generators
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Important for correct evaluation
)

# Verify label mapping
print("\nClass indices mapping:")
print(train_generator.class_indices)
print("\nSample file-label pairs:")
for idx, fname in enumerate(train_generator.filenames[:10]):
    print(f"{fname} -> {train_generator.labels[idx]}")

class_names = list(train_generator.class_indices.keys())
print("\nClass names in order:", class_names)

# Calculate class weights for imbalanced data
class_weights = compute_class_weight('balanced', 
                                   classes=np.unique(train_generator.classes), 
                                   y=train_generator.classes)
class_weight_dict = dict(enumerate(class_weights))
print("\nClass weights for imbalance:", class_weight_dict)

# Model builder function for Keras Tuner
def build_model(hp):
    base_model = InceptionV3(include_top=False, 
                           weights='imagenet', 
                           input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = False  # Freeze base model initially

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(hp.Int('units', 256, 512, step=128), activation='relu')(x)
    x = layers.Dropout(hp.Float('dropout', 0.3, 0.5, step=0.1))(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=Adam(hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5])),  # Lower learning rates
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    return model

# Keras Tuner setup
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='hyperparam_tuning',
    project_name='tea_leaf_disease_v2'
)

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    min_delta=0.001
)

# Search for best hyperparameters (phase 1 - feature extraction)
print("\nStarting hyperparameter search...")
tuner.search(
    train_generator,
    validation_data=val_generator,
    epochs=10,  # Fewer epochs for initial search
    callbacks=[early_stopping],
    class_weight=class_weight_dict
)

# Get best hyperparameters
best_hps = tuner.get_best_hyperparameters(1)[0]
print(f"""
Best Hyperparameters:
- Units: {best_hps.get('units')}
- Dropout: {best_hps.get('dropout')}
- Learning Rate: {best_hps.get('learning_rate')}
""")

# Build final model with best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Callbacks for full training
log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ModelCheckpoint('best_model_phase1.h5', monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=1e-7),
    TensorBoard(log_dir=log_dir)
]

# Phase 1: Train only the new layers (feature extraction)
print("\nPhase 1: Training top layers...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# Phase 2: Fine-tune the top layers of the base model
print("\nPhase 2: Fine-tuning top inception layers...")
base_model = model.layers[1]
base_model.trainable = True

# Freeze all layers except the last 40
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=best_hps.get('learning_rate') * 0.1),  # Lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Update callbacks for phase 2
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model_phase2.h5', monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7),
    TensorBoard(log_dir=log_dir)
]

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    initial_epoch=history.epoch[-1] + 1,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# Plot training history
def plot_training_history(history, fine_history=None):
    plt.figure(figsize=(12, 10))
    
    # Accuracy plot
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc (Phase 1)')
    plt.plot(history.history['val_accuracy'], label='Val Acc (Phase 1)')
    if fine_history:
        plt.plot(fine_history.history['accuracy'], label='Train Acc (Phase 2)')
        plt.plot(fine_history.history['val_accuracy'], label='Val Acc (Phase 2)')
    plt.title('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss (Phase 1)')
    plt.plot(history.history['val_loss'], label='Val Loss (Phase 1)')
    if fine_history:
        plt.plot(fine_history.history['loss'], label='Train Loss (Phase 2)')
        plt.plot(fine_history.history['val_loss'], label='Val Loss (Phase 2)')
    plt.title('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history, history_fine)

# Evaluation functions
def evaluate_model(model, generator):
    generator.reset()
    y_true = generator.classes
    y_pred = model.predict(generator)
    y_pred = np.argmax(y_pred, axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Per-class accuracy
    class_accuracy = 100 * cm.diagonal() / cm.sum(axis=1)
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {class_accuracy[i]:.2f}%")

# Evaluate on validation set
print("\nEvaluating model on validation set...")
evaluate_model(model, val_generator)

# Save final model
final_model_path = f"tea_leaf_disease_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
model.save(final_model_path)
print(f"\nModel saved as: {final_model_path}")