"""
PROJECT 1: MNIST Digit Recognition
===================================
This is your first CNN! We'll build a network that recognizes handwritten digits (0-9).

GOAL: Understand how a basic CNN works without worrying about complex math.

WHAT YOU'LL BUILD:
- Input: 28x28 grayscale images of handwritten digits
- Output: Prediction of which digit (0-9) it is
- Expected accuracy: 98-99%
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Set random seed for reproducibility (so you get same results each time)
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("PROJECT 1: MNIST Digit Recognition")
print("=" * 60)

# ============================================================================
# STEP 1: LOAD THE DATA
# ============================================================================
print("\n📂 Step 1: Loading MNIST dataset...")

# MNIST comes built into TensorFlow - super convenient!
# It contains 60,000 training images and 10,000 test images
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"✓ Training images: {x_train.shape[0]}")
print(f"✓ Test images: {x_test.shape[0]}")
print(f"✓ Image size: {x_train.shape[1]}x{x_train.shape[2]} pixels")

# ============================================================================
# STEP 2: VISUALIZE SOME EXAMPLES
# ============================================================================
print("\n🖼️  Step 2: Visualizing sample digits...")

# Let's see what we're working with!
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f'Label: {y_train[i]}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('project_1_mnist/sample_digits.png')
print("✓ Saved sample digits to 'sample_digits.png'")
plt.close()

# ============================================================================
# STEP 3: PREPROCESS THE DATA
# ============================================================================
print("\n🔧 Step 3: Preprocessing data...")

# Normalize pixel values from 0-255 to 0-1
# WHY? Neural networks work better with smaller numbers
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape images to add a "channel" dimension
# Shape changes from (28, 28) to (28, 28, 1)
# WHY? CNNs expect images in format (height, width, channels)
# Grayscale = 1 channel, RGB = 3 channels
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(f"✓ Training data shape: {x_train.shape}")
print(f"✓ Test data shape: {x_test.shape}")
print(f"✓ Pixel values normalized to range: [{x_train.min()}, {x_train.max()}]")

# ============================================================================
# STEP 4: BUILD THE CNN MODEL
# ============================================================================
print("\n🏗️  Step 4: Building the CNN architecture...")

model = keras.Sequential([
    # -------------------------
    # LAYER 1: First Convolutional Layer
    # -------------------------
    # What it does: Scans the image looking for simple patterns (edges, curves)
    # - 32 filters: Creates 32 different "feature detectors"
    # - (3,3) kernel: Each filter is a 3x3 window that slides across the image
    # - ReLU: Activation function that helps learn complex patterns
    # - Input shape: Tells the network our images are 28x28 with 1 color channel
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    
    # -------------------------
    # LAYER 2: First Pooling Layer
    # -------------------------
    # What it does: Shrinks the image by taking the maximum value in each 2x2 region
    # Why: Reduces computation and helps the network focus on important features
    # Effect: Image size reduces from 26x26 to 13x13
    keras.layers.MaxPooling2D((2, 2)),
    
    # -------------------------
    # LAYER 3: Second Convolutional Layer
    # -------------------------
    # What it does: Looks for more complex patterns using the features from layer 1
    # - 64 filters: More filters = can detect more complex patterns
    # - Builds on top of simple patterns to find more sophisticated features
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    
    # -------------------------
    # LAYER 4: Second Pooling Layer
    # -------------------------
    # Another pooling layer to further reduce size
    # Image size reduces from 11x11 to 5x5
    keras.layers.MaxPooling2D((2, 2)),
    
    # -------------------------
    # LAYER 5: Third Convolutional Layer
    # -------------------------
    # Even more complex patterns! Now we're detecting things like:
    # - Loops (for 0, 6, 8, 9)
    # - Vertical lines (for 1)
    # - Curves in specific positions (for 2, 3, 5)
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    
    # -------------------------
    # LAYER 6: Flatten
    # -------------------------
    # What it does: Converts the 2D feature maps into a 1D vector
    # Why: Dense layers (next) need 1D input
    # Think of it as: Unrolling a matrix into a single line
    keras.layers.Flatten(),
    
    # -------------------------
    # LAYER 7: Dense (Fully Connected) Layer
    # -------------------------
    # What it does: Combines all the detected features to make decisions
    # - 64 neurons: Each neuron looks at all features and learns patterns
    # Think of it as: The "thinking" layer that processes all the clues
    keras.layers.Dense(64, activation='relu'),
    
    # -------------------------
    # LAYER 8: Output Layer
    # -------------------------
    # What it does: Makes the final prediction
    # - 10 neurons: One for each digit (0-9)
    # - Softmax: Converts outputs to probabilities that sum to 1
    # Example output: [0.01, 0.02, 0.85, 0.03, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01]
    #                  This means the network is 85% confident it's a "2"
    keras.layers.Dense(10, activation='softmax')
])

# Display the model architecture
print("\n📋 Model Architecture:")
print("=" * 60)
model.summary()
print("=" * 60)

# Let's count the parameters
total_params = model.count_params()
print(f"\n✓ Total parameters: {total_params:,}")
print("   (Don't worry about the exact number - just know more parameters")
print("    means the model can learn more complex patterns!)")

# ============================================================================
# STEP 5: COMPILE THE MODEL
# ============================================================================
print("\n⚙️  Step 5: Compiling the model...")

# Compilation sets up HOW the model will learn:
# - Optimizer: 'adam' is a smart learning algorithm (it adjusts learning speed)
# - Loss: 'sparse_categorical_crossentropy' measures how wrong predictions are
# - Metrics: We track 'accuracy' (% of correct predictions)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled and ready to train!")
print("  - Optimizer: Adam (adaptive learning)")
print("  - Loss function: Sparse Categorical Crossentropy")
print("  - Tracking: Accuracy")

# ============================================================================
# STEP 6: TRAIN THE MODEL
# ============================================================================
print("\n🎯 Step 6: Training the model...")
print("This will take a few minutes. Watch the accuracy increase!\n")

# Training is where the magic happens!
# The model will:
# 1. Look at images
# 2. Make predictions
# 3. Check if predictions are correct
# 4. Adjust its parameters to improve
# 5. Repeat!

# Epochs: How many times to go through the entire dataset
# Batch size: How many images to look at before updating parameters
# Validation split: Use 10% of training data to check progress during training

history = model.fit(
    x_train, y_train,
    epochs=5,  # 5 passes through the data (you can increase for better accuracy)
    batch_size=128,
    validation_split=0.1,  # Use 10% of data for validation
    verbose=1  # Show progress
)

print("\n✓ Training complete!")

# ============================================================================
# STEP 7: EVALUATE ON TEST DATA
# ============================================================================
print("\n📊 Step 7: Evaluating on test data...")
print("Let's see how well it works on images it has NEVER seen before!\n")

# Test the model on completely new data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("=" * 60)
print(f"🎉 TEST ACCURACY: {test_accuracy * 100:.2f}%")
print("=" * 60)

if test_accuracy > 0.98:
    print("🌟 EXCELLENT! That's better than 98%!")
elif test_accuracy > 0.95:
    print("👍 GREAT! That's a solid result!")
else:
    print("👌 GOOD! With more training, you can improve this!")

# ============================================================================
# STEP 8: VISUALIZE TRAINING PROGRESS
# ============================================================================
print("\n📈 Step 8: Visualizing training progress...")

# Plot how accuracy improved over time
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
plt.title('Model Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='s')
plt.title('Model Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('project_1_mnist/training_progress.png')
print("✓ Saved training progress to 'training_progress.png'")
plt.close()

# ============================================================================
# STEP 9: MAKE PREDICTIONS ON NEW IMAGES
# ============================================================================
print("\n🔮 Step 9: Making predictions...")

# Let's test the model on some random images
num_samples = 10
indices = np.random.choice(len(x_test), num_samples, replace=False)

predictions = model.predict(x_test[indices], verbose=0)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = y_test[indices]

# Visualize predictions
plt.figure(figsize=(15, 3))
for i in range(num_samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[indices[i]].reshape(28, 28), cmap='gray')
    
    # Color the title green if correct, red if wrong
    color = 'green' if predicted_labels[i] == true_labels[i] else 'red'
    confidence = predictions[i][predicted_labels[i]] * 100
    
    plt.title(f'Pred: {predicted_labels[i]} ({confidence:.0f}%)\nTrue: {true_labels[i]}',
              color=color, fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig('project_1_mnist/predictions.png')
print("✓ Saved predictions to 'predictions.png'")
plt.close()

# ============================================================================
# STEP 10: CONFUSION MATRIX
# ============================================================================
print("\n🎯 Step 10: Creating confusion matrix...")

# Make predictions on all test images
all_predictions = model.predict(x_test, verbose=0)
predicted_classes = np.argmax(all_predictions, axis=1)

# Create confusion matrix
# This shows which digits the model confuses with each other
cm = confusion_matrix(y_test, predicted_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
plt.title('Confusion Matrix\n(Shows which digits are confused with each other)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('project_1_mnist/confusion_matrix.png')
print("✓ Saved confusion matrix to 'confusion_matrix.png'")
plt.close()

# ============================================================================
# STEP 11: SAVE THE MODEL
# ============================================================================
print("\n💾 Step 11: Saving the model...")

model.save('project_1_mnist/mnist_model.h5')
print("✓ Model saved as 'mnist_model.h5'")
print("  You can load it later with: model = keras.models.load_model('mnist_model.h5')")

# ============================================================================
# SUMMARY AND NEXT STEPS
# ============================================================================
print("\n" + "=" * 60)
print("🎊 CONGRATULATIONS! You've completed Project 1!")
print("=" * 60)
print("\n📚 What you learned:")
print("  ✓ How to load and preprocess image data")
print("  ✓ How to build a CNN with Conv2D and MaxPooling layers")
print("  ✓ How to train a model and monitor its progress")
print("  ✓ How to evaluate model performance")
print("  ✓ How to visualize predictions and errors")

print("\n🧪 Things to experiment with:")
print("  1. Change the number of epochs (line 187) - try 10 or 15")
print("  2. Add more filters to Conv2D layers - try 32, 64, 128")
print("  3. Add a Dropout layer after Dense layer to reduce overfitting")
print("  4. Change the optimizer to 'sgd' instead of 'adam'")
print("  5. Try different batch sizes (64, 256)")

print("\n➡️  Next Project: Fashion MNIST (project_2_fashion_mnist)")
print("   Learn to classify clothing items!")
print("\n" + "=" * 60)
