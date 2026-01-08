# Project 4: Cats vs Dogs Classifier 🐱🐶

## Overview
Binary classification meets DATA AUGMENTATION! This project teaches you how to artificially expand your dataset by transforming images, a crucial technique for real-world applications where data is limited.

## What You'll Learn
- ✅ **Binary classification** (2 classes vs 10 classes)
- ✅ **Data augmentation** - Artificially expand your dataset
- ✅ **Sigmoid activation** for binary output
- ✅ **Binary crossentropy** loss
- ✅ **Precision & Recall** metrics
- ✅ Working with higher resolution images (150x150)
- ✅ Real-world dataset structure

## Dataset
- **Name:** Cats vs Dogs (demonstration with synthetic data)
- **Real dataset:** Kaggle Cats vs Dogs (25,000 images)
- **Image format:** 150x150 RGB color
- **Classes:** 2 (Binary classification)
  - Cat (0)
  - Dog (1)

**Note:** This script uses synthetic demo data for learning. Instructions for real dataset included!

## Binary vs Multi-Class Classification

### Multi-Class (Projects 1-3)
```python
# Output layer
Dense(10, activation='softmax')

# Loss
loss='categorical_crossentropy'

# Output example: [0.05, 0.10, 0.75, 0.05, ...]
# "75% confident it's class 2"
```

### Binary (This Project)
```python
# Output layer
Dense(1, activation='sigmoid')

# Loss  
loss='binary_crossentropy'

# Output example: 0.85
# "85% dog, 15% cat"
```

**Why different?**
- Binary: Single probability (0-1)
- Multi-class: Multiple probabilities that sum to 1
- Same concepts, simpler math!

## How to Run

```bash
cd project_4_cats_dogs
python cats_dogs_cnn.py
```

## Expected Output (Demo Data)
- Training accuracy: ~50-60% (synthetic data)
- Validation accuracy: ~50-60% (synthetic data)
- Test accuracy: ~50-60% (synthetic data)
- Training time: ~3-5 minutes (CPU)

**With Real Data:** 80-85% accuracy possible!

## Files Generated
1. `augmentation_demo.png` - **IMPORTANT!** Shows augmentation effects
2. `training_metrics.png` - Accuracy, Loss, Precision, Recall curves
3. `confusion_matrix.png` - Binary confusion (2x2)
4. `cats_dogs_model.h5` - Trained model

## Model Architecture

```
Input (150x150x3)  ← Higher resolution!
    ↓
Data Augmentation Layer (RandomFlip, Rotate, Zoom, etc.)
    ↓
Rescaling (0-255 → 0-1)
    ↓
Conv2D(32) → BatchNorm → MaxPool → Dropout(0.2)
    ↓
Conv2D(64) → BatchNorm → MaxPool → Dropout(0.3)
    ↓
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.4)
    ↓
Conv2D(256) → BatchNorm → MaxPool → Dropout(0.4)
    ↓
Flatten → Dense(512) → BatchNorm → Dropout(0.5)
    ↓
Dense(128) → Dropout(0.5)
    ↓
Dense(1) → Sigmoid  ← Binary output!
```

## 🌟 DATA AUGMENTATION - The Key Concept!

### What is Data Augmentation?

**Problem:** You have 1,000 cat photos. Not enough for great accuracy.

**Solution:** Create MORE training examples by modifying existing ones!

### Transformations Applied

#### 1. **Random Horizontal Flip**
```python
RandomFlip("horizontal")
```
- Flips image left-to-right
- A cat looking left → now looking right
- **Still a cat!** Label unchanged
- Doubles your effective dataset

#### 2. **Random Rotation**
```python
RandomRotation(0.1)  # ±10% = ±36 degrees
```
- Rotates image slightly
- Teaches "cat-ness" is independent of angle
- Helps with photos taken at different angles

#### 3. **Random Zoom**
```python
RandomZoom(0.1)  # Zoom in/out by ±10%
```
- Simulates different distances
- Teaches: "Cat up close is still a cat"
- Handles scale variation

#### 4. **Random Translation**
```python
RandomTranslation(0.1, 0.1)  # Shift ±10%
```
- Moves image up/down/left/right
- Teaches: Position doesn't matter
- Object in corner? Still recognizable!

#### 5. **Random Brightness**
```python
RandomBrightness(0.2)  # ±20% brightness
```
- Simulates different lighting
- Indoor vs outdoor photos
- Shadows, overexposure, etc.

### Why It Works

**Key insight:** A cat is still a cat when:
- Flipped horizontally ✅
- Rotated slightly ✅
- Zoomed in/out ✅
- Shifted position ✅
- Different lighting ✅

**Result:** From 1,000 images → effectively 10,000+ variations!

### When Augmentation Happens

```python
# During TRAINING only
augmented = data_augmentation(image, training=True)

# During TESTING - NO augmentation
prediction = model(test_image, training=False)
```

**Why?**
- Training: Want variety and challenge
- Testing: Want consistent, fair evaluation

### Visual Example

Check `augmentation_demo.png` to see the same image transformed 9 different ways!

## Precision, Recall, and F1-Score

### Understanding the Metrics

#### Accuracy
```
Accuracy = (Correct Predictions) / (Total Predictions)
```
Simple but can be misleading with imbalanced data!

#### Precision
```
Precision = True Positives / (True Positives + False Positives)
```
**"Of all predicted dogs, how many were actually dogs?"**

High precision = Few false alarms

#### Recall
```
Recall = True Positives / (True Positives + False Negatives)
```
**"Of all actual dogs, how many did we find?"**

High recall = Few missed dogs

#### F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
**Harmonic mean** - Balances precision and recall

### Real-World Example

**Scenario:** Disease detection
- **High precision needed:** Don't want false positives (unnecessary worry)
- **High recall needed:** Don't want to miss actual cases

**Scenario:** Spam detection
- **High precision:** Non-spam shouldn't go to spam
- **Recall less critical:** Missing some spam is okay

### Confusion Matrix (Binary)

```
                Predicted
                Cat    Dog
Actual  Cat     TN     FP
        Dog     FN     TP

TN = True Negative  (Correctly predicted cat)
TP = True Positive  (Correctly predicted dog)
FN = False Negative (Dog predicted as cat)
FP = False Positive (Cat predicted as dog)
```

## Using Real Cats vs Dogs Dataset

### Step 1: Download Data
1. Go to [Kaggle Cats vs Dogs](https://www.kaggle.com/c/dogs-vs-cats/data)
2. Download `train.zip` (25,000 images)
3. Extract to your project folder

### Step 2: Organize Structure
```
project_4_cats_dogs/
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
```

### Step 3: Load Real Data
```python
import tensorflow as tf

# Training data
train_ds = tf.keras.utils.image_dataset_from_directory(
    'data/train',
    image_size=(150, 150),
    batch_size=32,
    label_mode='binary',
    validation_split=0.2,
    subset='training',
    seed=42
)

# Validation data
val_ds = tf.keras.utils.image_dataset_from_directory(
    'data/train',
    image_size=(150, 150),
    batch_size=32,
    label_mode='binary',
    validation_split=0.2,
    subset='validation',
    seed=42
)

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks
)
```

### Step 4: Expected Results
- Training accuracy: 85-90%
- Validation accuracy: 80-85%
- Test accuracy: 80-85%
- Training time: 10-15 minutes per epoch

## Troubleshooting

**Accuracy around 50% (random guessing)?**
- Using synthetic data - this is expected
- Try real Kaggle dataset for better results
- Check model is actually training (loss should decrease)

**Overfitting with real data?**
- Increase augmentation strength
- Add more dropout
- Use smaller model
- Get more training data

**Training too slow?**
- Reduce image size to 128x128 or 96x96
- Reduce batch size to 16
- Use fewer filters in Conv layers

**Out of memory?**
- Reduce batch size to 16 or 8
- Reduce image size
- Close other applications

## Experiments to Try

### 1. **Adjust Augmentation Strength**
```python
# More aggressive
layers.RandomRotation(0.3),  # ±30% = ±108 degrees
layers.RandomZoom(0.3),      # ±30%

# More conservative
layers.RandomRotation(0.05),  # ±5%
layers.RandomZoom(0.05),      # ±5%
```

### 2. **Try Vertical Flip**
```python
# Cats/dogs can be upside down in photos!
layers.RandomFlip("horizontal_and_vertical")
```

### 3. **Add More Augmentation Types**
```python
# Color adjustments
layers.RandomContrast(0.2),
layers.RandomBrightness(0.3),

# Advanced
layers.RandomCrop(height=140, width=140),
```

### 4. **Change Image Resolution**
```python
# Higher resolution (slower but potentially better)
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Lower resolution (faster but may lose detail)
IMG_HEIGHT, IMG_WIDTH = 96, 96
```

### 5. **Transfer Learning** (Advanced)
```python
# Use pre-trained VGG16
base_model = keras.applications.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(150, 150, 3)
)
base_model.trainable = False

# Add your classifier
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
```

## Understanding Training Behavior

### With Augmentation
**What you'll see:**
- Training accuracy increases slowly
- Validation accuracy close to training
- Small gap between train/val curves
- Less overfitting

### Without Augmentation
**What you'd see:**
- Training accuracy shoots up fast (90%+)
- Validation accuracy plateaus low (60-70%)
- Large gap between curves
- Severe overfitting

**Augmentation makes training harder but results better!**

## Key Differences from CIFAR-10

| Aspect | CIFAR-10 | Cats vs Dogs |
|--------|----------|--------------|
| Classes | 10 | 2 |
| Activation | Softmax | Sigmoid |
| Loss | Categorical | Binary |
| Resolution | 32x32 | 150x150 |
| Augmentation | Optional | Critical |
| Real-world | Medium | High |

## Next Steps

1. **Download real data** from Kaggle
2. **Implement loading** from directories
3. **Train on real images** - see 80%+ accuracy!
4. **Experiment** with different augmentation strategies
5. **Try transfer learning** with VGG16 or ResNet50
6. **Move to Project 5** - Design your own architectures!

## Key Takeaways

💡 **Data augmentation** is crucial with limited data
💡 **Binary classification** uses sigmoid + binary crossentropy
💡 **Precision & Recall** provide deeper insights than accuracy
💡 **Higher resolution** needs more computation but captures more detail
💡 **Real photos** are much harder than curated datasets
💡 **Augmentation during training only** - not during testing!

## Questions to Ponder

1. Why does augmentation help prevent overfitting?
2. What augmentations would hurt accuracy? (e.g., vertical flip for street signs)
3. Why use binary crossentropy instead of categorical?
4. How would you handle imbalanced data (90% cats, 10% dogs)?
5. What if you had 3 classes (cats, dogs, birds)?

---

👉 **Next Project:** [Custom CNN Architectures](../project_5_custom/) - Design your own!

You're almost done with the series! 🎉
