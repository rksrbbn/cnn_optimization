import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

# ==================== KONFIGURASI ====================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Path dataset
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/validation'
TEST_DIR = 'dataset/test'

# ==================== DATA AUGMENTATION ====================
print(" Mempersiapkan Data Augmentation...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.15,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f" Jumlah kelas: {num_classes}")
print(f" Kelas: {train_generator.class_indices}")

# ==================== MODEL DENGAN TRANSFER LEARNING ====================
print("\nMembangun Model dengan Transfer Learning (MobileNetV2)...")

def create_optimized_model(num_classes, img_size=224):
    """
    Model CNN dengan Transfer Learning, Dropout, dan Batch Normalization
    """
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build model
    model = models.Sequential([
        # Base model
        base_model,
        
        # Custom layers
        layers.GlobalAveragePooling2D(),
        
        # Dense Block 1
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Dense Block 2
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

model, base_model = create_optimized_model(num_classes, IMG_SIZE)
model.summary()

# ==================== COMPILE MODEL ====================
print("\n Kompilasi Model...")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

# ==================== CALLBACKS ====================
callbacks = [
    # Early stopping
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Save best model
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ==================== TRAINING PHASE 1: FROZEN BASE ====================
print("\n FASE 1: Training dengan Base Model Frozen...")

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# ==================== FINE-TUNING ====================
print("\n FASE 2: Fine-tuning (Unfreeze beberapa layers)...")

# Unfreeze top layers dari base model
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30  # Unfreeze 30 layers terakhir

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile dengan learning rate lebih kecil
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

# Continue training
history_fine = model.fit(
    train_generator,
    epochs=EPOCHS,
    initial_epoch=len(history.history['loss']),
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# ==================== EVALUASI FINAL ====================
print("\n Evaluasi Model pada Test Set...")

test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator)
f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)

print(f"\n{'='*50}")
print(f" HASIL EVALUASI FINAL:")
print(f"{'='*50}")
print(f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")
print(f"F1-Score:       {f1_score:.4f}")
print(f"{'='*50}\n")

# ==================== SAVE MODEL ====================
model.save('optimized_cnn_model.h5')
print(" Model disimpan sebagai 'optimized_cnn_model.h5'")

# Save class labels
import json
class_labels = {v: k for k, v in train_generator.class_indices.items()}
with open('class_labels.json', 'w') as f:
    json.dump(class_labels, f)
print(" Class labels disimpan sebagai 'class_labels.json'")

# ==================== PLOT TRAINING HISTORY ====================
def plot_history(history, history_fine=None):
    """Plot training history"""
    plt.figure(figsize=(15, 5))
    
    # Combine histories
    if history_fine:
        acc = history.history['accuracy'] + history_fine.history['accuracy']
        val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
        loss = history.history['loss'] + history_fine.history['loss']
        val_loss = history.history['val_loss'] + history_fine.history['val_loss']
    else:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print(" Plot training history disimpan sebagai 'training_history.png'")
    plt.show()

plot_history(history, history_fine)

print("\n Training selesai!")