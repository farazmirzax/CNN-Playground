# Project 2: Fashion MNIST Classifier 👕

## Overview
Level up from digits to clothing! This project teaches you to classify 10 different types of clothing and accessories using a more sophisticated CNN architecture.

## What You'll Learn
- ✅ Working with more complex visual patterns
- ✅ **Dropout layers** - Prevent overfitting
- ✅ **Batch Normalization** - Stabilize training
- ✅ **Callbacks** - Early stopping & learning rate scheduling
- ✅ Deeper network architectures
- ✅ Detailed performance analysis

## Dataset
- **Name:** Fashion MNIST
- **Size:** 60,000 training images, 10,000 test images
- **Image format:** 28x28 grayscale
- **Classes:** 10 clothing types
  - T-shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle boot

## How to Run

```bash
cd project_2_fashion_mnist
python fashion_mnist_cnn.py
```

## Expected Output
- Training accuracy: ~92-95%
- Validation accuracy: ~90-92%
- Test accuracy: ~88-92%
- Training time: ~3-5 minutes (CPU)
- May stop early if accuracy plateaus

## Files Generated
1. `sample_items.png` - One example from each class
2. `class_variation.png` - 20 different T-shirts (shows variety)
3. `training_history.png` - Accuracy and loss curves
4. `confusion_matrix.png` - Which items get confused
5. `correct_predictions.png` - Successfully classified items
6. `incorrect_predictions.png` - Mistakes (learn from these!)
7. `fashion_mnist_model.h5` - Saved trained model

## Model Architecture

```
Input (28x28x1)
    ↓
Conv2D (32) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D (64) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D (128) → BatchNorm
    ↓
Flatten
    ↓
Dense (128) → Dropout(0.5)
    ↓
Dense (10) → Softmax
```

## New Concepts Explained

### 🎯 Dropout
**What it does:** Randomly "turns off" neurons during training

**Why it helps:**
- Prevents the network from memorizing training data
- Forces learning of redundant representations
- Like studying by teaching others - you learn better!

**How it works:**
```python
Dropout(0.5)  # Drops 50% of connections randomly
```

During training: Some neurons are ignored
During testing: All neurons active (no dropout)

**Think of it as:** Practicing basketball with one hand tied - makes you more versatile!

### 📊 Batch Normalization
**What it does:** Normalizes layer outputs during training

**Why it helps:**
- Stabilizes training (less sensitive to initialization)
- Allows higher learning rates
- Acts as mild regularization
- Faster convergence

**How it works:**
```python
BatchNormalization()  # Add after Conv2D or Dense layers
```

**Think of it as:** Keeping the playing field level so everyone plays well together

### 📞 Callbacks
**What they do:** Functions that run during training to monitor/control the process

**Early Stopping:**
```python
EarlyStopping(patience=3)  # Stop if no improvement for 3 epochs
```
- Prevents wasting time when model stops improving
- Automatically restores best weights

**Reduce Learning Rate:**
```python
ReduceLROnPlateau(factor=0.5, patience=2)
```
- Reduces learning rate when stuck
- Helps escape plateaus and find better solutions

**Think of it as:** A smart coach that adjusts training based on performance

## Why Fashion MNIST is Harder Than MNIST

1. **Visual Complexity**
   - Clothing has more variation than digits
   - Same item can look very different (angles, styles)
   - More subtle differences between classes

2. **Similar Classes**
   - T-shirt vs Shirt vs Pullover (all upper body)
   - Sandal vs Sneaker vs Ankle boot (all footwear)
   - Coat vs Pullover (both outerwear)

3. **Within-Class Variation**
   - T-shirts come in many styles
   - Bags have countless shapes
   - Each class has high diversity

## Per-Class Performance

Typically:
- **Easy classes (90%+):** Trouser, Bag, Sneaker
  - Very distinctive shapes
  
- **Medium classes (85-90%):** T-shirt, Pullover, Coat
  - Some overlap with similar items
  
- **Hard classes (80-85%):** Shirt, Sandal, Ankle boot
  - Often confused with similar classes

## Troubleshooting

**Accuracy not improving after ~85%?**
- This is normal! Fashion MNIST has natural limits
- Some items are genuinely hard to distinguish
- Try more epochs or different architecture

**Overfitting (train >> validation accuracy)?**
- Increase dropout rates (try 0.6 or 0.7)
- Add more data augmentation
- Reduce model complexity

**Training stops early?**
- Early stopping activated (good thing!)
- Model found optimal point
- Check `restore_best_weights=True` is working

**"ReduceLROnPlateau" message appears?**
- Learning rate was automatically reduced
- Helps fine-tune the model
- Completely normal and beneficial

## Experiments to Try

### 1. **Adjust Dropout Rates**
```python
# Original: 0.25, 0.25, 0.5
# Try: More aggressive
Dropout(0.4), Dropout(0.4), Dropout(0.6)

# Or: Less dropout
Dropout(0.1), Dropout(0.2), Dropout(0.3)
```

### 2. **Remove Batch Normalization**
Comment out all `BatchNormalization()` layers and see the difference

### 3. **Change Architecture Depth**
```python
# Add another Conv block
keras.layers.Conv2D(256, (3, 3), activation='relu'),
keras.layers.BatchNormalization(),
```

### 4. **Modify Callback Patience**
```python
# More patient (trains longer)
EarlyStopping(patience=5)

# Less patient (stops sooner)
EarlyStopping(patience=2)
```

### 5. **Different Optimizers**
```python
# Instead of Adam, try:
optimizer='sgd'  # Classic
optimizer=keras.optimizers.RMSprop(learning_rate=0.001)
optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
```

## Understanding the Results

### Confusion Matrix Insights
Look for patterns:
- **Diagonal (high values):** Correct predictions ✅
- **Off-diagonal (high values):** Common confusions ⚠️

**Common confusions:**
- Shirt ↔ T-shirt (both upper body wear)
- Pullover ↔ Coat (both outerwear)
- Sandal ↔ Sneaker (both footwear)

### Training Curves
**Healthy training:**
- Both train & val accuracy increase together
- Small gap between them
- Smooth curves

**Overfitting signs:**
- Train accuracy high (95%+)
- Val accuracy low (80-85%)
- Large gap between curves

**Underfitting signs:**
- Both accuracies low
- Not improving much
- Need more capacity (layers/filters)

### Classification Report
- **Precision:** Of predicted T-shirts, how many were actually T-shirts?
- **Recall:** Of all actual T-shirts, how many did we find?
- **F1-score:** Harmonic mean of precision and recall

## Comparing to Project 1

| Aspect | MNIST (Project 1) | Fashion MNIST (Project 2) |
|--------|-------------------|---------------------------|
| Accuracy | 98-99% | 88-92% |
| Difficulty | Easy | Medium |
| Patterns | Simple (digits) | Complex (clothing) |
| Architecture | Basic | Improved |
| Techniques | None | Dropout, BatchNorm, Callbacks |
| Training Time | 2 min | 3-5 min |

## Next Steps

Once comfortable with this project:

1. **Experiment** - Try all suggested modifications
2. **Analyze** - Study which classes are hardest and why
3. **Compare** - Run with/without Dropout to see the effect
4. **Move on** - Ready for Project 3: CIFAR-10 (COLOR images!)

## Key Takeaways

💡 **Dropout prevents overfitting** by forcing redundancy
💡 **Batch Normalization stabilizes training** and allows faster learning
💡 **Callbacks automate monitoring** and save you time
💡 **Not all datasets are equal** - Fashion MNIST is inherently harder
💡 **88-92% is excellent** for this dataset - don't expect 98%!

## Questions to Ponder

1. Why does Dropout help during training but not testing?
2. What happens if you put BatchNorm BEFORE activation instead of after?
3. Why are some clothing items easier to classify than others?
4. How would you improve accuracy further?

---

👉 **Next Project:** [CIFAR-10 Classification](../project_3_cifar10/) - Color images await!

Happy learning! 🚀
