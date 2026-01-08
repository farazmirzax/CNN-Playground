"""
PROJECT 4: Cats vs Dogs Classifier with Data Augmentation
==========================================================
Binary classification challenge! But with a twist - DATA AUGMENTATION!

What's new:
- Binary classification (cat vs dog, not 10 classes)
- Data augmentation (artificially expand dataset)
- Transfer learning concepts
- Working with real photos (not curated datasets)

Note: This uses a subset of Cats vs Dogs data built into Keras
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("PROJECT 4: Cats vs Dogs Classifier")
print("=" * 60)

# ============================================================================
# STEP 1: PREPARE DATA
# ============================================================================
print("\n📂 Step 1: Preparing dataset...")
print("\nNote: We'll create synthetic data for demonstration.")
print("In real projects, you'd download the Kaggle Cats vs Dogs dataset.")

# For this example, we'll use a simple approach with augmentation
# In practice, you'd load real cat/dog images from a directory

# Let's create a simple dataset generator
# (In real scenario, use tf.keras.utils.image_dataset_from_directory)

print("✓ Dataset preparation explained (see code comments)")

# ============================================================================
# WHAT IS DATA AUGMENTATION?
# ============================================================================
print("\n💡 CONCEPT: Data Augmentation")
print("=" * 60)
print("Problem: Not enough training data? Photos all look the same?")
print("\nSolution: CREATE more training examples by modifying existing ones!")
print("\nTransformations we'll use:")
print("  🔄 Rotation: Rotate images slightly")
print("  ↔️  Horizontal flip: Mirror the image")
print("  🔍 Zoom: Zoom in/out randomly")
print("  ⬆️  Shift: Move image up/down/left/right")
print("  💡 Brightness: Adjust lighting")
print("\nWhy it works:")
print("  • A cat photo is still a cat when flipped!")
print("  • A dog at different angles is still a dog!")
print("  • Helps model learn what matters (features)")
print("  • Prevents overfitting (memorizing training data)")
print("=" * 60)

# ============================================================================
# STEP 2: BUILD DATA AUGMENTATION PIPELINE
# ============================================================================
print("\n🔧 Step 2: Building data augmentation pipeline...")

# Create augmentation layers
# These apply random transformations during training
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),  # Flip left-right
    layers.RandomRotation(0.1),  # Rotate ±10%
    layers.RandomZoom(0.1),  # Zoom ±10%
    layers.RandomTranslation(0.1, 0.1),  # Shift ±10%
    layers.RandomBrightness(0.2),  # Brightness ±20%
], name="data_augmentation")

print("✓ Augmentation pipeline created!")
print("  Each training image will be randomly transformed")

# Visualize augmentation effect
print("\n🖼️  Demonstrating augmentation effects...")

# Create a sample image (checkerboard pattern for visualization)
sample_img = np.zeros((150, 150, 3))
sample_img[25:125, 25:75] = [1, 0, 0]  # Red square
sample_img[25:125, 75:125] = [0, 1, 0]  # Green square
sample_img = sample_img[np.newaxis, ...]  # Add batch dimension

# Apply augmentation multiple times
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

axes[0].imshow(sample_img[0])
axes[0].set_title('Original', fontweight='bold')
axes[0].axis('off')

for i in range(1, 10):
    augmented = data_augmentation(sample_img, training=True)
    axes[i].imshow(augmented[0])
    axes[i].set_title(f'Augmented {i}')
    axes[i].axis('off')

plt.suptitle('Data Augmentation: Same image, different transformations!', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('project_4_cats_dogs/augmentation_demo.png')
print("✓ Saved augmentation demo to 'augmentation_demo.png'")
plt.close()

# ============================================================================
# STEP 3: BUILD CNN WITH AUGMENTATION
# ============================================================================
print("\n🏗️  Step 3: Building CNN with integrated augmentation...")

# Input shape for higher resolution images
IMG_HEIGHT, IMG_WIDTH = 150, 150

model = keras.Sequential([
    # Augmentation happens first (only during training!)
    data_augmentation,
    
    # Rescaling
    layers.Rescaling(1./255),
    
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),
    
    # Block 4
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    
    # Binary output (cat or dog)
    # Using sigmoid because it's binary classification!
    layers.Dense(1, activation='sigmoid')  # Output between 0 and 1
])

# Build the model
model.build(input_shape=(None, IMG_HEIGHT, IMG_WIDTH, 3))

print("\n📋 Model Architecture:")
print("=" * 60)
model.summary()
print("=" * 60)

# ============================================================================
# BINARY VS MULTI-CLASS
# ============================================================================
print("\n💡 Binary Classification vs Multi-class:")
print("=" * 60)
print("Multi-class (Projects 1-3):")
print("  • Softmax activation (probabilities sum to 1)")
print("  • Categorical crossentropy loss")
print("  • Example: [0.1, 0.05, 0.7, 0.15] for 4 classes")
print("\nBinary (This project):")
print("  • Sigmoid activation (single probability)")
print("  • Binary crossentropy loss")
print("  • Example: 0.85 means 85% dog, 15% cat")
print("  • Simpler but same principles!")
print("=" * 60)

# ============================================================================
# STEP 4: COMPILE FOR BINARY CLASSIFICATION
# ============================================================================
print("\n⚙️  Step 4: Compiling for binary classification...")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR
    loss='binary_crossentropy',  # For binary classification
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("✓ Model compiled!")
print("  • Loss: Binary Crossentropy")
print("  • Metrics: Accuracy, Precision, Recall")

print("\n💡 Understanding Precision & Recall:")
print("  Precision: Of all predicted dogs, how many were actually dogs?")
print("  Recall: Of all actual dogs, how many did we find?")
print("  Both are important for balanced performance!")

# ============================================================================
# STEP 5: DEMONSTRATION WITH SYNTHETIC DATA
# ============================================================================
print("\n🎯 Step 5: Creating synthetic demo data...")

# For demonstration, create simple synthetic data
# In real project, use: tf.keras.utils.image_dataset_from_directory()

def create_demo_data(n_samples=1000, img_size=150):
    """Create synthetic cat/dog images for demonstration"""
    x_data = np.random.rand(n_samples, img_size, img_size, 3)
    y_data = np.random.randint(0, 2, n_samples).astype('float32')
    
    # Add some patterns to make it slightly realistic
    for i in range(n_samples):
        if y_data[i] == 0:  # Cat
            x_data[i, :, :, 0] += 0.3  # More red
        else:  # Dog
            x_data[i, :, :, 2] += 0.3  # More blue
    
    x_data = np.clip(x_data, 0, 1)
    return x_data, y_data

print("Creating synthetic training data...")
x_train_demo, y_train_demo = create_demo_data(2000, IMG_HEIGHT)
x_val_demo, y_val_demo = create_demo_data(400, IMG_HEIGHT)
x_test_demo, y_test_demo = create_demo_data(400, IMG_HEIGHT)

print(f"✓ Training samples: {len(x_train_demo)}")
print(f"✓ Validation samples: {len(x_val_demo)}")
print(f"✓ Test samples: {len(x_test_demo)}")

# ============================================================================
# STEP 6: TRAIN WITH AUGMENTATION
# ============================================================================
print("\n🎯 Step 6: Training with data augmentation...")
print("This demonstrates the training process.\n")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.000001
    )
]

history = model.fit(
    x_train_demo, y_train_demo,
    epochs=15,
    batch_size=32,
    validation_data=(x_val_demo, y_val_demo),
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Training complete!")

# ============================================================================
# STEP 7: EVALUATE
# ============================================================================
print("\n📊 Step 7: Evaluating model...")

test_results = model.evaluate(x_test_demo, y_test_demo, verbose=0)
test_loss, test_acc, test_precision, test_recall = test_results

print("\n" + "=" * 60)
print("📈 TEST RESULTS:")
print("=" * 60)
print(f"  Accuracy:  {test_acc * 100:.2f}%")
print(f"  Precision: {test_precision * 100:.2f}%")
print(f"  Recall:    {test_recall * 100:.2f}%")
print(f"  Loss:      {test_loss:.4f}")
print("=" * 60)

# ============================================================================
# STEP 8: VISUALIZE TRAINING WITH AUGMENTATION
# ============================================================================
print("\n📈 Step 8: Visualizing training progress...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Training', marker='o')
axes[0, 0].plot(history.history['val_accuracy'], label='Validation', marker='s')
axes[0, 0].set_title('Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Training', marker='o')
axes[0, 1].plot(history.history['val_loss'], label='Validation', marker='s')
axes[0, 1].set_title('Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(history.history['precision'], label='Training', marker='o')
axes[1, 0].plot(history.history['val_precision'], label='Validation', marker='s')
axes[1, 0].set_title('Precision')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Recall
axes[1, 1].plot(history.history['recall'], label='Training', marker='o')
axes[1, 1].plot(history.history['val_recall'], label='Validation', marker='s')
axes[1, 1].set_title('Recall')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('project_4_cats_dogs/training_metrics.png')
print("✓ Saved training metrics")
plt.close()

# ============================================================================
# STEP 9: PREDICTIONS AND CONFUSION MATRIX
# ============================================================================
print("\n🔮 Step 9: Analyzing predictions...")

y_pred_prob = model.predict(x_test_demo, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Confusion matrix for binary classification
cm = confusion_matrix(y_test_demo, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Cat', 'Dog'],
            yticklabels=['Cat', 'Dog'],
            square=True, cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix\n(Binary Classification)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('project_4_cats_dogs/confusion_matrix.png')
print("✓ Saved confusion matrix")
plt.close()

print("\nClassification Report:")
print("=" * 60)
print(classification_report(y_test_demo, y_pred, target_names=['Cat', 'Dog']))

# ============================================================================
# STEP 10: SAVE MODEL
# ============================================================================
print("\n💾 Step 10: Saving model...")

model.save('project_4_cats_dogs/cats_dogs_model.h5')
print("✓ Model saved!")

# ============================================================================
# REAL-WORLD APPLICATION GUIDE
# ============================================================================
print("\n" + "=" * 60)
print("🌍 REAL-WORLD APPLICATION GUIDE")
print("=" * 60)

print("\n📥 To use real Cats vs Dogs dataset:")
print("""
1. Download from Kaggle:
   https://www.kaggle.com/c/dogs-vs-cats/data

2. Organize in folders:
   data/
     train/
       cats/
         cat.0.jpg
         cat.1.jpg
         ...
       dogs/
         dog.0.jpg
         dog.1.jpg
         ...
     validation/
       cats/
       dogs/

3. Load with:
   train_ds = tf.keras.utils.image_dataset_from_directory(
       'data/train',
       image_size=(150, 150),
       batch_size=32,
       label_mode='binary'
   )

4. Train:
   model.fit(train_ds, validation_data=val_ds, epochs=30)
""")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("🎊 CONGRATULATIONS! Project 4 Complete!")
print("=" * 60)

print("\n📚 What you learned:")
print("  ✓ Data augmentation techniques")
print("  ✓ Binary classification (vs multi-class)")
print("  ✓ Sigmoid activation & binary crossentropy")
print("  ✓ Precision & Recall metrics")
print("  ✓ Working with higher resolution images")
print("  ✓ How to handle real-world image datasets")

print("\n🎓 Key takeaways:")
print("  • Augmentation multiplies your effective dataset size")
print("  • Binary classification is simpler but uses same concepts")
print("  • Precision and Recall give deeper insights than accuracy alone")
print("  • Real photos need more preprocessing than curated datasets")

print("\n🧪 Experiments for real dataset:")
print("  1. Download actual Cats vs Dogs data from Kaggle")
print("  2. Try different augmentation parameters")
print("  3. Experiment with image sizes (180x180, 224x224)")
print("  4. Try transfer learning (VGG16, ResNet50)")
print("  5. Add more augmentation types (contrast, saturation)")

print("\n💡 Pro tips:")
print("  • Data augmentation is crucial with limited data")
print("  • Higher resolution = better accuracy but slower training")
print("  • Augmentation only happens during training, not testing")
print("  • Balance your dataset (equal cats and dogs)")

print("\n➡️  Next Project: Custom CNN Architecture (project_5_custom)")
print("   Design your own CNN from scratch!")
print("\n" + "=" * 60)
