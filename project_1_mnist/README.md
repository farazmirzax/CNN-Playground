# Project 1: MNIST Digit Recognition 🔢

## Overview
Your first CNN project! Build a neural network that recognizes handwritten digits (0-9).

## What You'll Learn
- ✅ Basic CNN architecture
- ✅ Convolutional and pooling layers
- ✅ Data preprocessing
- ✅ Training and evaluation
- ✅ Visualization of results

## Dataset
- **Name:** MNIST
- **Size:** 60,000 training images, 10,000 test images
- **Image format:** 28x28 grayscale
- **Classes:** 10 digits (0-9)

## How to Run

```bash
python mnist_cnn.py
```

## Expected Output
- Training accuracy: ~99%
- Test accuracy: ~98-99%
- Training time: ~2-3 minutes (CPU)

## Files Generated
1. `sample_digits.png` - Examples from the dataset
2. `training_progress.png` - Accuracy and loss curves
3. `predictions.png` - Sample predictions with confidence scores
4. `confusion_matrix.png` - Which digits get confused
5. `mnist_model.h5` - Saved trained model

## Model Architecture

```
Input (28x28x1)
    ↓
Conv2D (32 filters, 3x3) → ReLU
    ↓
MaxPooling (2x2)
    ↓
Conv2D (64 filters, 3x3) → ReLU
    ↓
MaxPooling (2x2)
    ↓
Conv2D (64 filters, 3x3) → ReLU
    ↓
Flatten
    ↓
Dense (64 neurons) → ReLU
    ↓
Dense (10 neurons) → Softmax
    ↓
Output (10 classes)
```

## Key Concepts Explained

### Convolutional Layer (Conv2D)
Think of it as a feature detector that slides across the image looking for patterns:
- **First layer:** Detects simple features (edges, curves)
- **Deeper layers:** Detect complex features (loops, specific shapes)

### Pooling Layer (MaxPooling)
Reduces image size while keeping important information:
- Helps the network focus on "what" is present, not "where" exactly
- Makes computation faster
- Reduces overfitting

### Dense Layer
The "decision maker" that combines all detected features:
- Takes all the features found by Conv layers
- Learns which combination of features = which digit

## Troubleshooting

**Low accuracy (<90%)?**
- Run for more epochs (try 10)
- Check if data is properly normalized
- Make sure you reshaped images correctly

**Training too slow?**
- Reduce batch size
- Reduce number of filters
- Use fewer epochs for testing

**"Out of memory" error?**
- Reduce batch size to 64 or 32
- Close other applications

## Experiments to Try

1. **Add Dropout:** Prevent overfitting
   ```python
   keras.layers.Dropout(0.5)  # Add after Dense layer
   ```

2. **More epochs:** Train longer
   ```python
   epochs=10  # Instead of 5
   ```

3. **Different optimizers:**
   ```python
   optimizer='sgd'  # Instead of 'adam'
   ```

4. **Deeper network:** Add more Conv layers
   ```python
   keras.layers.Conv2D(128, (3, 3), activation='relu')
   ```

## Understanding the Results

### Confusion Matrix
- **Diagonal values (high):** Correctly classified
- **Off-diagonal values:** Confusions
- Common confusions: 4↔9, 3↔8, 7↔1

### Training vs Validation Accuracy
- **Both increasing:** Model is learning well ✅
- **Training high, validation low:** Overfitting ⚠️
- **Both low:** Underfitting (need more complexity) ⚠️

## Next Steps
Once you're comfortable with this project, move on to:
👉 **Project 2: Fashion MNIST** (More challenging patterns!)

## Questions to Ponder
1. Why do we use multiple Conv layers instead of just one?
2. What happens if you remove the pooling layers?
3. Why normalize pixel values to 0-1?
4. What does the first Conv layer learn vs the last Conv layer?

Happy learning! 🚀
