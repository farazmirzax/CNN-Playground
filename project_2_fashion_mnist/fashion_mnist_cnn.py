"""
PROJECT 2: Fashion MNIST Classifier
====================================
Now let's tackle something more challenging! Instead of digits, we'll classify clothing items.

This project introduces:
- More complex visual patterns
- Dropout (prevents overfitting)
- Better architecture design
- Callback functions for monitoring

DATASET: 10 types of clothing/accessories
- T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("PROJECT 2: Fashion MNIST Classifier")
print("=" * 60)

# Class names for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================
print("\n📂 Step 1: Loading Fashion MNIST dataset...")

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

print(f"✓ Training images: {x_train.shape[0]}")
print(f"✓ Test images: {x_test.shape[0]}")
print(f"✓ Image size: {x_train.shape[1]}x{x_train.shape[2]} pixels")
print(f"✓ Number of classes: {len(class_names)}")

# ============================================================================
# STEP 2: VISUALIZE THE DATASET
# ============================================================================
print("\n🖼️  Step 2: Visualizing sample items...")

# Show one example from each class
plt.figure(figsize=(15, 6))
for i in range(10):
    # Find first occurrence of each class
    idx = np.where(y_train == i)[0][0]
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[idx], cmap='gray')
    plt.title(f'{i}: {class_names[i]}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('project_2_fashion_mnist/sample_items.png')
print("✓ Saved sample items to 'sample_items.png'")
plt.close()

# Show multiple examples of one class to see variation
print("\n📸 Visualizing variation within a class (T-shirts)...")
tshirt_indices = np.where(y_train == 0)[0][:20]
plt.figure(figsize=(15, 3))
for i, idx in enumerate(tshirt_indices):
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_train[idx], cmap='gray')
    plt.axis('off')
plt.suptitle('Notice how different T-shirts can look!')
plt.tight_layout()
plt.savefig('project_2_fashion_mnist/class_variation.png')
print("✓ Saved class variation to 'class_variation.png'")
plt.close()

# ============================================================================
# STEP 3: PREPROCESS DATA
# ============================================================================
print("\n🔧 Step 3: Preprocessing data...")

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(f"✓ Data shape: {x_train.shape}")
print(f"✓ Pixel range: [{x_train.min():.1f}, {x_train.max():.1f}]")

# ============================================================================
# STEP 4: BUILD IMPROVED CNN MODEL
# ============================================================================
print("\n🏗️  Step 4: Building an improved CNN architecture...")
print("Key improvements from Project 1:")
print("  • Added Dropout layers (reduce overfitting)")
print("  • More filters in deeper layers")
print("  • BatchNormalization (helps training stability)")

model = keras.Sequential([
    # Block 1: Initial feature extraction
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.BatchNormalization(),  # NEW! Normalizes activations
    keras.layers.MaxPooling2D((2, 2)),
    
    # Block 2: More complex features
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    
    # Block 3: High-level features
    keras.layers.Conv2D(128, (3, 3), activation='relu'),  # More filters!
    keras.layers.BatchNormalization(),
    
    # Flatten and classify
    keras.layers.Flatten(),
    
    # Dense layers with Dropout
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),  # NEW! Randomly drops 50% of connections during training
                                 # This prevents overfitting by forcing the network
                                 # to learn redundant representations
    
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),  # Less dropout in second layer
    
    # Output layer
    keras.layers.Dense(10, activation='softmax')
])

print("\n📋 Model Architecture:")
print("=" * 60)
model.summary()
print("=" * 60)

# ============================================================================
# WHAT IS DROPOUT?
# ============================================================================
print("\n💡 Understanding Dropout:")
print("  Dropout is like practicing with a blindfold on!")
print("  - During training: Randomly 'turns off' some neurons")
print("  - Forces the network to learn multiple ways to recognize patterns")
print("  - Prevents relying too heavily on any single feature")
print("  - During testing: All neurons are active (no dropout)")
print("\n  Think of it as: If you study for an exam by teaching others,")
print("  you learn the material better than just reading it yourself!")

# ============================================================================
# STEP 5: COMPILE WITH LEARNING RATE SCHEDULE
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
print("\n📞 Step 6: Setting up training callbacks...")

# Callbacks are functions that run during training
# They help monitor and control the training process

# 1. Early Stopping: Stop if validation accuracy doesn't improve
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',  # Watch validation accuracy
    patience=3,  # Wait 3 epochs before stopping
    restore_best_weights=True,  # Use the best model, not the last one
    verbose=1
)

# 2. Reduce learning rate when stuck
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Reduce learning rate by half
    patience=2,  # Wait 2 epochs
    min_lr=0.00001,
    verbose=1
)

print("✓ Callbacks configured:")
print("  • Early stopping (prevents wasting time)")
print("  • Learning rate reduction (helps escape plateaus)")

# ============================================================================
# STEP 7: TRAIN THE MODEL
# ============================================================================
print("\n🎯 Step 7: Training the model...")
print("This may take 3-5 minutes. Watch the progress!\n")

history = model.fit(
    x_train, y_train,
    epochs=20,  # We can try more epochs thanks to early stopping
    batch_size=128,
    validation_split=0.2,  # Use 20% for validation
    callbacks=[early_stop, reduce_lr],
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

if test_accuracy > 0.91:
    print("🌟 OUTSTANDING! You've built an excellent classifier!")
elif test_accuracy > 0.88:
    print("👍 GREAT JOB! That's a strong result!")
else:
    print("👌 GOOD START! Try adjusting hyperparameters to improve!")

# ============================================================================
# STEP 9: DETAILED PERFORMANCE ANALYSIS
# ============================================================================
print("\n🔍 Step 9: Analyzing performance per class...")

# Get predictions
y_pred = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("\nClassification Report:")
print("=" * 60)
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# Find best and worst performing classes
print("\n📈 Per-class accuracy:")
for i, class_name in enumerate(class_names):
    class_mask = y_test == i
    class_accuracy = np.mean(y_pred_classes[class_mask] == y_test[class_mask])
    emoji = "🟢" if class_accuracy > 0.90 else "🟡" if class_accuracy > 0.80 else "🔴"
    print(f"  {emoji} {class_name:15s}: {class_accuracy*100:5.1f}%")

# ============================================================================
# STEP 10: VISUALIZE TRAINING HISTORY
# ============================================================================
print("\n📈 Step 10: Visualizing training history...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Training', marker='o')
axes[0].plot(history.history['val_accuracy'], label='Validation', marker='s')
axes[0].set_title('Model Accuracy Over Time')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Training', marker='o')
axes[1].plot(history.history['val_loss'], label='Validation', marker='s')
axes[1].set_title('Model Loss Over Time')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('project_2_fashion_mnist/training_history.png')
print("✓ Saved training history to 'training_history.png'")
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
plt.title('Confusion Matrix\nWhich items are confused with each other?')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('project_2_fashion_mnist/confusion_matrix.png')
print("✓ Saved confusion matrix to 'confusion_matrix.png'")
plt.close()

# Analyze most confused pairs
print("\n🔀 Most common confusions:")
cm_copy = cm.copy()
np.fill_diagonal(cm_copy, 0)  # Remove diagonal (correct predictions)
for _ in range(3):
    i, j = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
    print(f"  • {class_names[i]} mistaken for {class_names[j]}: {cm_copy[i,j]} times")
    cm_copy[i, j] = 0

# ============================================================================
# STEP 12: VISUALIZE PREDICTIONS
# ============================================================================
print("\n🔮 Step 12: Visualizing predictions...")

# Show correct predictions
correct_indices = np.where(y_pred_classes == y_test)[0]
correct_samples = np.random.choice(correct_indices, 10, replace=False)

plt.figure(figsize=(15, 3))
for i, idx in enumerate(correct_samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    confidence = y_pred[idx][y_pred_classes[idx]] * 100
    plt.title(f'✓ {class_names[y_test[idx]]}\n{confidence:.0f}% confident', 
              color='green', fontsize=9)
    plt.axis('off')
plt.suptitle('Correct Predictions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('project_2_fashion_mnist/correct_predictions.png')
print("✓ Saved correct predictions to 'correct_predictions.png'")
plt.close()

# Show incorrect predictions (these are interesting!)
incorrect_indices = np.where(y_pred_classes != y_test)[0]
if len(incorrect_indices) > 0:
    incorrect_samples = np.random.choice(incorrect_indices, min(10, len(incorrect_indices)), replace=False)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(incorrect_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        confidence = y_pred[idx][y_pred_classes[idx]] * 100
        plt.title(f'Pred: {class_names[y_pred_classes[idx]]}\nTrue: {class_names[y_test[idx]]}\n{confidence:.0f}% confident',
                  color='red', fontsize=8)
        plt.axis('off')
    plt.suptitle('Incorrect Predictions (Learn from mistakes!)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('project_2_fashion_mnist/incorrect_predictions.png')
    print("✓ Saved incorrect predictions to 'incorrect_predictions.png'")
    plt.close()

# ============================================================================
# STEP 13: SAVE THE MODEL
# ============================================================================
print("\n💾 Step 13: Saving the model...")

model.save('project_2_fashion_mnist/fashion_mnist_model.h5')
print("✓ Model saved successfully!")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("🎊 CONGRATULATIONS! Project 2 Complete!")
print("=" * 60)

print("\n📚 New concepts you learned:")
print("  ✓ Dropout - prevents overfitting")
print("  ✓ BatchNormalization - stabilizes training")
print("  ✓ Callbacks - early stopping & learning rate scheduling")
print("  ✓ Deeper networks with more filters")
print("  ✓ Detailed performance analysis")

print("\n🎓 Key insights:")
print("  • Fashion items are harder than digits!")
print("  • Some classes are naturally more confusable")
print("  • Dropout helps when you have limited data")
print("  • Early stopping prevents wasting time")

print("\n🧪 Experiments to try:")
print("  1. Remove Dropout - see how accuracy changes")
print("  2. Try different dropout rates (0.3, 0.7)")
print("  3. Add more Conv2D layers")
print("  4. Try different filter sizes (5x5 instead of 3x3)")
print("  5. Use data augmentation (rotation, flipping)")

print("\n➡️  Next Project: CIFAR-10 (project_3_cifar10)")
print("   Ready for COLOR images with 10 different objects!")
print("\n" + "=" * 60)
