import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import os
from pathlib import Path

# ==================== KONFIGURASI ====================
MODEL_PATH = 'optimized_cnn_model.h5'
CLASS_LABELS_PATH = 'class_labels.json'
TEST_IMAGE_DIR = 'test_images'  # Folder dengan gambar baru untuk testing
IMG_SIZE = 224

# ==================== LOAD MODEL ====================
print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

with open(CLASS_LABELS_PATH, 'r') as f:
    class_labels = json.load(f)

print(f"📋 Classes: {list(class_labels.values())}")

# ==================== HELPER FUNCTIONS ====================
def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess single image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

def predict_single_image(model, image_path, class_labels):
    """Predict single image"""
    img_array, original_img = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = class_labels[str(predicted_class_idx)]
    
    return predicted_class, confidence, predictions[0], original_img

# ==================== TEST 1: GAMBAR BARU ====================
def test_new_images(test_dir):
    """Test model dengan gambar baru yang belum pernah dilihat"""
    print("\n" + "="*60)
    print("🧪 TEST 1: GAMBAR BARU (UNSEEN DATA)")
    print("="*60)
    
    if not os.path.exists(test_dir):
        print(f"⚠️ Folder '{test_dir}' tidak ditemukan!")
        print(f"💡 Buat folder '{test_dir}' dan masukkan gambar untuk testing")
        return
    
    image_files = list(Path(test_dir).glob('*.*'))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    if len(image_files) == 0:
        print(f"⚠️ Tidak ada gambar di folder '{test_dir}'")
        return
    
    print(f"📁 Ditemukan {len(image_files)} gambar untuk testing\n")
    
    results = []
    
    # Create visualization
    n_images = min(len(image_files), 12)  # Max 12 images
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(image_files[:n_images]):
        predicted_class, confidence, all_probs, img = predict_single_image(
            model, img_path, class_labels
        )
        
        results.append({
            'image': img_path.name,
            'prediction': predicted_class,
            'confidence': confidence
        })
        
        # Display
        row = idx // n_cols
        col = idx % n_cols
        
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        
        color = 'green' if confidence > 0.8 else 'orange' if confidence > 0.5 else 'red'
        axes[row, col].set_title(
            f"{predicted_class}\n{confidence:.2%}",
            fontsize=10,
            color=color,
            weight='bold'
        )
        
        print(f"✅ {img_path.name}")
        print(f"   Prediksi: {predicted_class}")
        print(f"   Confidence: {confidence:.2%}")
        print()
    
    # Hide empty subplots
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_results_new_images.png', dpi=300, bbox_inches='tight')
    print("💾 Hasil disimpan ke 'test_results_new_images.png'")
    plt.show()
    
    return results

# ==================== TEST 2: BATCH PREDICTION ====================
def test_batch_prediction(test_data_dir):
    """Test dengan dataset terstruktur (dengan label ground truth)"""
    print("\n" + "="*60)
    print("🧪 TEST 2: BATCH PREDICTION & CONFUSION MATRIX")
    print("="*60)
    
    if not os.path.exists(test_data_dir):
        print(f"⚠️ Folder '{test_data_dir}' tidak ditemukan!")
        return
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"📁 Total gambar test: {test_generator.samples}")
    print(f"📋 Classes: {test_generator.class_indices}\n")
    
    # Predict
    print("🔮 Melakukan prediksi...")
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Classification Report
    print("\n" + "="*60)
    print("📊 CLASSIFICATION REPORT")
    print("="*60)
    
    class_names = list(test_generator.class_indices.keys())
    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, weight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n💾 Confusion matrix disimpan ke 'confusion_matrix.png'")
    plt.show()
    
    # Per-class accuracy
    print("\n" + "="*60)
    print("📊 PER-CLASS ACCURACY")
    print("="*60)
    
    for i, class_name in enumerate(class_names):
        class_correct = cm[i, i]
        class_total = cm[i].sum()
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"{class_name:20s}: {class_acc:.2%} ({class_correct}/{class_total})")

# ==================== TEST 3: CONFIDENCE ANALYSIS ====================
def test_confidence_analysis(test_data_dir):
    """Analisis distribusi confidence scores"""
    print("\n" + "="*60)
    print("🧪 TEST 3: CONFIDENCE ANALYSIS")
    print("="*60)
    
    if not os.path.exists(test_data_dir):
        print(f"⚠️ Folder '{test_data_dir}' tidak ditemukan!")
        return
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    predictions = model.predict(test_generator, verbose=0)
    confidence_scores = np.max(predictions, axis=1)
    
    # Statistics
    print(f"\n📊 Confidence Statistics:")
    print(f"   Mean:   {np.mean(confidence_scores):.4f}")
    print(f"   Median: {np.median(confidence_scores):.4f}")
    print(f"   Std:    {np.std(confidence_scores):.4f}")
    print(f"   Min:    {np.min(confidence_scores):.4f}")
    print(f"   Max:    {np.max(confidence_scores):.4f}")
    
    # Distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(confidence_scores, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Confidence Scores')
    plt.axvline(np.mean(confidence_scores), color='red', linestyle='--', label='Mean')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    counts = [np.sum(confidence_scores >= t) for t in thresholds]
    percentages = [c / len(confidence_scores) * 100 for c in counts]
    
    bars = plt.bar([f'≥{t}' for t in thresholds], percentages, edgecolor='black')
    plt.ylabel('Percentage of Predictions (%)')
    plt.xlabel('Confidence Threshold')
    plt.title('Predictions Above Threshold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
    print("\n💾 Analisis disimpan ke 'confidence_analysis.png'")
    plt.show()

# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 MEMULAI TESTING & EVALUASI MODEL")
    print("="*60)
    
    # Test 1: Gambar baru (unseen)
    results = test_new_images(TEST_IMAGE_DIR)
    
    # Test 2: Batch prediction dengan ground truth
    # Ganti dengan path folder test Anda yang berisi subfolder per kelas
    test_batch_prediction('dataset/test')
    
    # Test 3: Confidence analysis
    test_confidence_analysis('dataset/test')
    
    print("\n" + "="*60)
    print("✅ TESTING SELESAI!")
    print("="*60)
    print("\n📁 File hasil yang dibuat:")
    print("   - test_results_new_images.png")
    print("   - confusion_matrix.png")
    print("   - confidence_analysis.png")
    print("\n🎉 Semua test berhasil dijalankan!")