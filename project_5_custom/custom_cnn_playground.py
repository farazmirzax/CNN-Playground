"""
PROJECT 5: Custom CNN Architecture - Your Playground!
=====================================================
This is YOUR project! Design, experiment, and learn by doing.

This project provides:
- A framework for experimenting with different architectures
- Tools for comparing model performance
- Guidelines for hyperparameter tuning
- Space to try YOUR ideas!

Mission: Build the best CNN you can and understand WHY it works!
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time

np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("PROJECT 5: Custom CNN Architecture - Design Your Own!")
print("=" * 70)

# ============================================================================
# CHOOSE YOUR DATASET
# ============================================================================
print("\n📂 Step 1: Choose your dataset")
print("=" * 70)
print("Available datasets:")
print("  1. MNIST - Handwritten digits (easy)")
print("  2. Fashion MNIST - Clothing items (medium)")
print("  3. CIFAR-10 - Color images (hard)")

# For this template, we'll use Fashion MNIST
# You can change this to experiment with different datasets!
DATASET_CHOICE = 2  # Change this: 1=MNIST, 2=Fashion, 3=CIFAR-10

if DATASET_CHOICE == 1:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    class_names = [str(i) for i in range(10)]
    img_shape = (28, 28, 1)
    dataset_name = "MNIST"
elif DATASET_CHOICE == 2:
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    img_shape = (28, 28, 1)
    dataset_name = "Fashion MNIST"
else:
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train, y_test = y_train.flatten(), y_test.flatten()
    class_names = ['airplane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    img_shape = (32, 32, 3)
    dataset_name = "CIFAR-10"

print(f"\n✓ Selected dataset: {dataset_name}")
print(f"  Training samples: {len(x_train)}")
print(f"  Test samples: {len(x_test)}")
print(f"  Image shape: {img_shape}")

# Preprocess
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

if len(x_train.shape) == 3:  # Add channel dimension for grayscale
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

# ============================================================================
# ARCHITECTURE FACTORY
# ============================================================================
print("\n🏗️  Step 2: Architecture Design Laboratory")
print("=" * 70)

def create_custom_cnn(architecture_name='balanced', img_shape=(28, 28, 1)):
    """
    Create different CNN architectures for experimentation
    
    Available architectures:
    - 'simple': Minimal CNN (fast, lower accuracy)
    - 'balanced': Good trade-off (Projects 1-2 style)
    - 'deep': Many layers (high capacity)
    - 'wide': Many filters per layer
    - 'residual': With skip connections
    """
    
    if architecture_name == 'simple':
        print("🔹 Building SIMPLE architecture...")
        print("   2 Conv blocks, minimal parameters")
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_shape),
            keras.layers.MaxPooling2D((2, 2)),
            
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
    
    elif architecture_name == 'balanced':
        print("🔹 Building BALANCED architecture...")
        print("   3 Conv blocks, dropout, batch norm")
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_shape),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])
    
    elif architecture_name == 'deep':
        print("🔹 Building DEEP architecture...")
        print("   5 Conv blocks, high capacity")
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=img_shape),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.2),
            
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.3),
            
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.4),
            
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])
    
    elif architecture_name == 'wide':
        print("🔹 Building WIDE architecture...")
        print("   3 Conv blocks, many filters per layer")
        model = keras.Sequential([
            keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=img_shape),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            
            keras.layers.Conv2D(256, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            
            keras.layers.Conv2D(512, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])
    
    else:  # residual
        print("🔹 Building architecture with RESIDUAL connections...")
        print("   Advanced: Skip connections for better gradient flow")
        
        inputs = keras.Input(shape=img_shape)
        
        # Block 1
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        
        # Residual connection
        residual = keras.layers.Conv2D(32, (1, 1), padding='same')(inputs)
        x = keras.layers.add([x, residual])
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        
        # Block 2
        residual = x
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        
        residual = keras.layers.Conv2D(64, (1, 1), padding='same')(residual)
        x = keras.layers.add([x, residual])
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        
        # Output
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# ============================================================================
# EXPERIMENT: COMPARE ARCHITECTURES
# ============================================================================
print("\n🧪 Step 3: Running architecture comparison experiment...")
print("=" * 70)

# Choose which architectures to compare (comment out to skip)
architectures_to_test = [
    'simple',
    'balanced',
    # 'deep',      # Uncomment to test
    # 'wide',      # Uncomment to test
    # 'residual',  # Uncomment to test
]

results = []

for arch_name in architectures_to_test:
    print(f"\n{'='*70}")
    print(f"Testing: {arch_name.upper()}")
    print(f"{'='*70}")
    
    # Create model
    model = create_custom_cnn(arch_name, img_shape)
    
    # Count parameters
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    total_params = model.count_params()
    print(f"📊 Total parameters: {total_params:,}")
    
    # Train
    print("\n🎯 Training...")
    start_time = time.time()
    
    history = model.fit(
        x_train, y_train,
        epochs=5,  # Quick training for comparison
        batch_size=128,
        validation_split=0.2,
        verbose=0
    )
    
    training_time = time.time() - start_time
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    # Store results
    results.append({
        'name': arch_name,
        'params': total_params,
        'train_acc': history.history['accuracy'][-1],
        'val_acc': history.history['val_accuracy'][-1],
        'test_acc': test_accuracy,
        'time': training_time,
        'history': history
    })
    
    print(f"✓ Training accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"✓ Validation accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
    print(f"✓ Test accuracy: {test_accuracy*100:.2f}%")
    print(f"✓ Training time: {training_time:.1f}s")

# ============================================================================
# COMPARE RESULTS
# ============================================================================
print("\n\n" + "="*70)
print("📊 COMPARISON RESULTS")
print("="*70)

# Create comparison table
print("\n{:<15} {:>12} {:>10} {:>10} {:>10} {:>8}".format(
    "Architecture", "Parameters", "Train Acc", "Val Acc", "Test Acc", "Time"
))
print("-" * 70)

for r in results:
    print("{:<15} {:>12,} {:>9.2f}% {:>9.2f}% {:>9.2f}% {:>7.1f}s".format(
        r['name'], r['params'], 
        r['train_acc']*100, r['val_acc']*100, r['test_acc']*100,
        r['time']
    ))

# Find best model
best_test = max(results, key=lambda x: x['test_acc'])
fastest = min(results, key=lambda x: x['time'])
most_efficient = min(results, key=lambda x: x['params'])

print("\n🏆 Best test accuracy: {} ({:.2f}%)".format(
    best_test['name'], best_test['test_acc']*100
))
print("⚡ Fastest training: {} ({:.1f}s)".format(
    fastest['name'], fastest['time']
))
print("💡 Most efficient: {} ({:,} params)".format(
    most_efficient['name'], most_efficient['params']
))

# ============================================================================
# VISUALIZE COMPARISON
# ============================================================================
print("\n📈 Generating comparison visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Test accuracy comparison
names = [r['name'] for r in results]
test_accs = [r['test_acc']*100 for r in results]
axes[0, 0].bar(names, test_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(names)])
axes[0, 0].set_title('Test Accuracy Comparison', fontweight='bold')
axes[0, 0].set_ylabel('Accuracy (%)')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Parameters vs Accuracy
params = [r['params']/1000 for r in results]  # In thousands
axes[0, 1].scatter(params, test_accs, s=200, alpha=0.6, c=range(len(names)), cmap='viridis')
for i, name in enumerate(names):
    axes[0, 1].annotate(name, (params[i], test_accs[i]), 
                        xytext=(5, 5), textcoords='offset points')
axes[0, 1].set_title('Parameters vs Accuracy', fontweight='bold')
axes[0, 1].set_xlabel('Parameters (thousands)')
axes[0, 1].set_ylabel('Test Accuracy (%)')
axes[0, 1].grid(True, alpha=0.3)

# Training time comparison
times = [r['time'] for r in results]
axes[1, 0].bar(names, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(names)])
axes[1, 0].set_title('Training Time Comparison', fontweight='bold')
axes[1, 0].set_ylabel('Time (seconds)')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Overfitting analysis (Train vs Val accuracy)
train_accs = [r['train_acc']*100 for r in results]
val_accs = [r['val_acc']*100 for r in results]
x_pos = np.arange(len(names))
width = 0.35
axes[1, 1].bar(x_pos - width/2, train_accs, width, label='Train', alpha=0.8)
axes[1, 1].bar(x_pos + width/2, val_accs, width, label='Validation', alpha=0.8)
axes[1, 1].set_title('Overfitting Analysis', fontweight='bold')
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(names)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('project_5_custom/architecture_comparison.png')
print("✓ Saved comparison to 'architecture_comparison.png'")
plt.close()

# ============================================================================
# TRAIN BEST MODEL FULLY
# ============================================================================
print("\n\n🎯 Training best architecture with full epochs...")
print("="*70)

best_arch = best_test['name']
print(f"Selected architecture: {best_arch}")

final_model = create_custom_cnn(best_arch, img_shape)
final_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

final_history = final_model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Final evaluation
final_test_loss, final_test_acc = final_model.evaluate(x_test, y_test, verbose=0)

print("\n" + "="*70)
print(f"🎉 FINAL TEST ACCURACY: {final_test_acc*100:.2f}%")
print("="*70)

# Save final model
final_model.save('project_5_custom/custom_cnn_model.h5')
print("\n✓ Final model saved!")

# ============================================================================
# SUMMARY AND INSIGHTS
# ============================================================================
print("\n\n" + "="*70)
print("🎊 CONGRATULATIONS! You've completed all 5 projects!")
print("="*70)

print("\n📚 Journey Summary:")
print("  Project 1: MNIST - Learned CNN basics")
print("  Project 2: Fashion - Added dropout & batch norm")
print("  Project 3: CIFAR-10 - Handled color images")
print("  Project 4: Cats vs Dogs - Mastered data augmentation")
print("  Project 5: Custom - Designed your own architectures!")

print("\n🎓 Key Insights from Architecture Comparison:")
print("  • Simple models train fast but may underperform")
print("  • More parameters ≠ always better (overfitting risk)")
print("  • Balanced architectures often win")
print("  • Training time scales with model complexity")
print("  • Regularization (dropout, batch norm) helps!")

print("\n💡 What you've mastered:")
print("  ✓ Building CNNs from scratch")
print("  ✓ Understanding conv, pooling, dense layers")
print("  ✓ Preventing overfitting")
print("  ✓ Data augmentation")
print("  ✓ Comparing architectures")
print("  ✓ Hyperparameter tuning")

print("\n🚀 Next Steps to Level Up:")
print("  1. Try Transfer Learning (VGG16, ResNet50)")
print("  2. Implement attention mechanisms")
print("  3. Work with real-world datasets")
print("  4. Deploy models (Flask, FastAPI)")
print("  5. Experiment with object detection (YOLO)")
print("  6. Try semantic segmentation (U-Net)")

print("\n🧪 Experiments to try in this project:")
print("  • Change DATASET_CHOICE (1, 2, or 3)")
print("  • Uncomment different architectures to test")
print("  • Modify architectures (add layers, change filters)")
print("  • Try different optimizers (SGD, RMSprop)")
print("  • Experiment with learning rates")
print("  • Add your own custom architecture!")

print("\n💪 You're now ready for your final year projects!")
print("   Remember: Understanding > Memorizing")
print("   Every expert started where you are now.")
print("\n" + "="*70)
