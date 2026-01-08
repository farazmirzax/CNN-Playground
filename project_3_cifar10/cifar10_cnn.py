"""
PROJECT 3: CIFAR-10 Image Classification
=========================================
Welcome to COLOR images! This is a big step up.

New challenges:
- Color images (RGB = 3 channels instead of 1)
- More complex objects (cars, airplanes, animals)
- Smaller images with less detail (32x32 pixels)
- More challenging classification task

DATASET: 10 classes
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("PROJECT 3: CIFAR-10 Image Classification")
print("=" * 60)

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ============================================================================
# STEP 1: LOAD CIFAR-10 DATA
# ============================================================================
print("\n📂 Step 1: Loading CIFAR-10 dataset...")

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Flatten labels (they come as 2D arrays)
y_train = y_train.flatten()
y_test = y_test.flatten()

print(f"✓ Training images: {x_train.shape[0]}")
print(f"✓ Test images: {x_test.shape[0]}")
print(f"✓ Image size: {x_train.shape[1]}x{x_train.shape[2]} pixels")
print(f"✓ Color channels: {x_train.shape[3]} (RGB)")
print(f"✓ Classes: {len(class_names)}")

# ============================================================================
# STEP 2: VISUALIZE COLOR IMAGES
# ============================================================================
print("\n🖼️  Step 2: Visualizing sample images...")

plt.figure(figsize=(15, 6))
for i in range(10):
    idx = np.where(y_train == i)[0][0]
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[idx])  # No cmap needed - it's color!
    plt.title(f'{class_names[i]}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('project_3_cifar10/sample_images.png', dpi=150)
print("✓ Saved sample images to 'sample_images.png'")
plt.close()

# Show variation in one class
print("\n📸 Showing variety in 'airplane' class...")
plane_indices = np.where(y_train == 0)[0][:20]
plt.figure(figsize=(15, 3))
for i, idx in enumerate(plane_indices):
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_train[idx])
    plt.axis('off')
plt.suptitle('Same class, very different appearances!')
plt.tight_layout()
plt.savefig('project_3_cifar10/class_variation.png', dpi=150)
print("✓ Saved class variation to 'class_variation.png'")
plt.close()

# ============================================================================
# STEP 3: PREPROCESS DATA
# ============================================================================
print("\n🔧 Step 3: Preprocessing data...")

# Normalize to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(f"✓ Training shape: {x_train.shape}")
print(f"✓ Test shape: {x_test.shape}")
print(f"✓ Pixel range: [{x_train.min():.1f}, {x_train.max():.1f}]")

# Calculate mean and std for normalization (optional but helps!)
mean = np.mean(x_train, axis=(0, 1, 2))
std = np.std(x_train, axis=(0, 1, 2))

print(f"✓ Dataset mean (RGB): {mean}")
print(f"✓ Dataset std (RGB): {std}")

# Optional: Apply standardization
# x_train = (x_train - mean) / (std + 1e-7)
# x_test = (x_test - mean) / (std + 1e-7)

# ============================================================================
# STEP 4: BUILD DEEPER CNN FOR COLOR IMAGES
# ============================================================================
print("\n🏗️  Step 4: Building deeper CNN for color images...")
print("\nKey changes for CIFAR-10:")
print("  • Input shape: (32, 32, 3) - 3 color channels!")
print("  • More convolutional layers (images are complex)")
print("  • Batch normalization after each Conv layer")
print("  • Multiple dropout layers")

model = keras.Sequential([
    # Block 1: Initial feature extraction
    keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', 
                        input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.2),
    
    # Block 2: More complex features
    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),
    
    # Block 3: High-level features
    keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.4),
    
    # Dense layers
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    
    # Output
    keras.layers.Dense(10, activation='softmax')
])

print("\n📋 Model Architecture:")
print("=" * 60)
model.summary()
print("=" * 60)

total_params = model.count_params()
print(f"\n✓ Total parameters: {total_params:,}")

# ============================================================================
# WHAT'S NEW?
# ============================================================================
print("\n💡 What's new in this architecture?")
print("\n1. padding='same':")
print("   - Keeps the image size constant after convolution")
print("   - Allows deeper networks without shrinking too much")
print("   - Better preserves spatial information")

print("\n2. Double Conv layers before pooling:")
print("   - Two Conv layers learn more complex patterns")
print("   - Common pattern: Conv-Conv-Pool")
print("   - Helps build hierarchical features")

print("\n3. Progressive dropout (0.2 → 0.3 → 0.4 → 0.5):")
print("   - Less dropout early (preserve learning)")
print("   - More dropout late (prevent overfitting)")
print("   - Adapts to network depth")

# ============================================================================
# STEP 5: COMPILE WITH DATA AUGMENTATION PREP
# ============================================================================
print("\n⚙️  Step 5: Compiling model...")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled!")

# ============================================================================
# STEP 6: SETUP CALLBACKS
# ============================================================================
print("\n📞 Step 6: Setting up callbacks...")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'project_3_cifar10/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
]

print("✓ Callbacks ready!")

# ============================================================================
# STEP 7: TRAIN THE MODEL
# ============================================================================
print("\n🎯 Step 7: Training the model...")
print("This will take 5-10 minutes. Color images are more complex!\n")

history = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,  # Smaller batch size for better generalization
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Training complete!")

# ============================================================================
# STEP 8: EVALUATE PERFORMANCE
# ============================================================================
print("\n📊 Step 8: Evaluating on test data...")

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("\n" + "=" * 60)
print(f"🎉 TEST ACCURACY: {test_accuracy * 100:.2f}%")
print("=" * 60)

if test_accuracy > 0.75:
    print("🌟 EXCELLENT! Color images are tough - great job!")
elif test_accuracy > 0.70:
    print("👍 VERY GOOD! That's a solid result for CIFAR-10!")
elif test_accuracy > 0.60:
    print("👌 GOOD! With more training, you can improve!")
else:
    print("📈 Keep experimenting - CIFAR-10 is challenging!")

# ============================================================================
# STEP 9: ANALYZE PER-CLASS PERFORMANCE
# ============================================================================
print("\n🔍 Step 9: Per-class analysis...")

y_pred = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print("=" * 60)
print(classification_report(y_test, y_pred_classes, target_names=class_names))

print("\n📈 Per-class accuracy:")
for i, class_name in enumerate(class_names):
    class_mask = y_test == i
    class_accuracy = np.mean(y_pred_classes[class_mask] == y_test[class_mask])
    
    if class_accuracy > 0.80:
        emoji = "🟢"
    elif class_accuracy > 0.65:
        emoji = "🟡"
    else:
        emoji = "🔴"
    
    print(f"  {emoji} {class_name:12s}: {class_accuracy*100:5.1f}%")

# ============================================================================
# STEP 10: TRAINING HISTORY VISUALIZATION
# ============================================================================
print("\n📈 Step 10: Visualizing training progress...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Training', marker='o', markersize=3)
axes[0].plot(history.history['val_accuracy'], label='Validation', marker='s', markersize=3)
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Training', marker='o', markersize=3)
axes[1].plot(history.history['val_loss'], label='Validation', marker='s', markersize=3)
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('project_3_cifar10/training_history.png')
print("✓ Saved training history")
plt.close()

# ============================================================================
# STEP 11: CONFUSION MATRIX
# ============================================================================
print("\n🎯 Step 11: Creating confusion matrix...")

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            square=True)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('project_3_cifar10/confusion_matrix.png')
print("✓ Saved confusion matrix")
plt.close()

# Find most confused pairs
print("\n🔀 Most common confusions:")
cm_copy = cm.copy()
np.fill_diagonal(cm_copy, 0)
for _ in range(5):
    i, j = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
    print(f"  • {class_names[i]} ← → {class_names[j]}: {cm_copy[i,j]} times")
    cm_copy[i, j] = 0

# ============================================================================
# STEP 12: VISUALIZE PREDICTIONS
# ============================================================================
print("\n🔮 Step 12: Visualizing predictions...")

# Correct predictions
correct_indices = np.where(y_pred_classes == y_test)[0]
correct_samples = np.random.choice(correct_indices, 15, replace=False)

plt.figure(figsize=(15, 6))
for i, idx in enumerate(correct_samples):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[idx])
    confidence = y_pred[idx][y_pred_classes[idx]] * 100
    plt.title(f'{class_names[y_test[idx]]}\n{confidence:.0f}%', 
              color='green', fontsize=9)
    plt.axis('off')
plt.suptitle('Correct Predictions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('project_3_cifar10/correct_predictions.png', dpi=150)
print("✓ Saved correct predictions")
plt.close()

# Incorrect predictions
incorrect_indices = np.where(y_pred_classes != y_test)[0]
if len(incorrect_indices) > 0:
    incorrect_samples = np.random.choice(incorrect_indices, min(15, len(incorrect_indices)), replace=False)
    
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(incorrect_samples):
        plt.subplot(3, 5, i + 1)
        plt.imshow(x_test[idx])
        confidence = y_pred[idx][y_pred_classes[idx]] * 100
        plt.title(f'P: {class_names[y_pred_classes[idx]]}\nT: {class_names[y_test[idx]]}\n{confidence:.0f}%',
                  color='red', fontsize=8)
        plt.axis('off')
    plt.suptitle('Incorrect Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('project_3_cifar10/incorrect_predictions.png', dpi=150)
    print("✓ Saved incorrect predictions")
    plt.close()

# ============================================================================
# STEP 13: SAVE MODEL
# ============================================================================
print("\n💾 Step 13: Saving final model...")

model.save('project_3_cifar10/cifar10_model.h5')
print("✓ Model saved!")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("🎊 CONGRATULATIONS! Project 3 Complete!")
print("=" * 60)

print("\n📚 What you learned:")
print("  ✓ Working with RGB (color) images")
print("  ✓ Deeper CNN architectures")
print("  ✓ padding='same' for maintaining spatial dimensions")
print("  ✓ Double Conv layers before pooling")
print("  ✓ Progressive dropout strategies")
print("  ✓ Model checkpointing")

print("\n🎓 Why CIFAR-10 is harder:")
print("  • Smaller images (32x32) = less detail")
print("  • More visual variety within classes")
print("  • Some classes look similar (cat vs dog, truck vs automobile)")
print("  • Real-world photos with different lighting, angles, backgrounds")

print("\n🧪 Experiments to try:")
print("  1. Add data augmentation (rotation, flipping, zoom)")
print("  2. Try transfer learning with pre-trained models")
print("  3. Increase image size with upsampling")
print("  4. Add more Conv layers")
print("  5. Try different optimizers (SGD with momentum)")

print("\n💡 Pro tips:")
print("  • Color images have 3x more data than grayscale")
print("  • Deeper networks can learn more complex features")
print("  • Overfitting is common - use dropout & augmentation")
print("  • Some classes are naturally harder (cats vs dogs)")

print("\n➡️  Next Project: Cats vs Dogs (project_4_cats_dogs)")
print("   Real-world images with data augmentation!")
print("\n" + "=" * 60)
