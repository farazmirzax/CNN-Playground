# Project 5: Custom CNN Architecture Playground 🎮

## Overview
**This is YOUR laboratory!** Design, test, and compare different CNN architectures. Learn what works, what doesn't, and most importantly - WHY. This project synthesizes everything you've learned and encourages experimentation.

## What You'll Learn
- ✅ **Architecture design principles**
- ✅ **Comparing multiple models** systematically
- ✅ **Trade-offs** - Accuracy vs Speed vs Parameters
- ✅ **Hyperparameter tuning** strategies
- ✅ **Model selection** based on requirements
- ✅ **Scientific experimentation** approach

## Project Goal

Build and compare different CNN architectures to understand:
1. **Simple** - Fast but potentially lower accuracy
2. **Balanced** - Good trade-off (like your previous projects)
3. **Deep** - Many layers, high capacity
4. **Wide** - Many filters per layer
5. **Residual** - Skip connections for better gradient flow

Then train the best one fully!

## How to Run

```bash
cd project_5_custom
python custom_cnn_playground.py
```

## Configuration

At the top of the script, you can choose:

```python
# Choose your dataset
DATASET_CHOICE = 2  # 1=MNIST, 2=Fashion MNIST, 3=CIFAR-10

# Choose which architectures to test
architectures_to_test = [
    'simple',
    'balanced',
    # 'deep',      # Uncomment to test
    # 'wide',      # Uncomment to test
    # 'residual',  # Uncomment to test
]
```

## Expected Output
- Quick comparison (5 epochs each)
- Architecture comparison table
- Best model selection
- Full training of best model (20 epochs)
- Training time: ~10-20 minutes total

## Files Generated
1. `architecture_comparison.png` - Compare all architectures
2. `custom_cnn_model.h5` - Best model fully trained

## Architecture Options Explained

### 1. 🔹 Simple Architecture
```python
Conv2D(32) → MaxPool
Conv2D(64) → MaxPool
Flatten → Dense(64) → Output
```

**Pros:**
- ⚡ Fast training
- 🪶 Few parameters
- 🎯 Good for easy datasets

**Cons:**
- 📉 Lower capacity
- ❌ May underfit complex data
- 🚫 Less feature learning

**When to use:** Quick prototypes, simple datasets, limited compute

### 2. 🔹 Balanced Architecture
```python
Conv(32) → BN → MaxPool → Dropout(0.25)
Conv(64) → BN → MaxPool → Dropout(0.25)
Conv(128) → BN
Flatten → Dense(128) → Dropout(0.5) → Output
```

**Pros:**
- ⚖️ Great trade-off
- 🎯 Good accuracy
- 🛡️ Prevents overfitting
- ⏱️ Reasonable speed

**Cons:**
- 🤷 Not the absolute best at anything
- 💭 Middle-of-the-road performance

**When to use:** Most projects, balanced requirements, general purpose

### 3. 🔹 Deep Architecture
```python
Conv(32) → Conv(32) → MaxPool → Dropout
Conv(64) → Conv(64) → MaxPool → Dropout
Conv(128) → Conv(128) → MaxPool → Dropout
Flatten → Dense(256) → Dropout → Output
```

**Pros:**
- 🎯 High capacity
- 📈 Best accuracy potential
- 🧠 Learns hierarchical features
- 🏆 State-of-the-art approach

**Cons:**
- 🐌 Slow training
- 💾 Many parameters
- ⚠️ Overfitting risk
- 🔌 Needs more data

**When to use:** Complex datasets, accuracy priority, have compute power

### 4. 🔹 Wide Architecture
```python
Conv(128) → BN → MaxPool
Conv(256) → BN → MaxPool
Conv(512) → BN
Flatten → Dense(256) → Dropout → Output
```

**Pros:**
- 💪 High capacity per layer
- 🎯 Can learn rich features
- 📊 Good for diverse data

**Cons:**
- 💾 MANY parameters
- 🐌 Slow training
- ⚠️ Severe overfitting risk
- 🔌 Needs LOTS of data

**When to use:** Large datasets, feature-rich problems, powerful hardware

### 5. 🔹 Residual Architecture
```python
x → Conv → Conv → (+) → Output
    ↓____________↑
    Skip connection
```

**Pros:**
- 🌟 Advanced technique
- 📈 Better gradient flow
- 🎯 Can be very deep
- 🏆 Used in ResNet, etc.

**Cons:**
- 🧩 More complex
- 💭 Harder to understand
- ⚙️ Tricky to implement
- 🎚️ More hyperparameters

**When to use:** Very deep networks, gradient problems, research projects

## Understanding the Comparison

### Metrics Compared

#### 1. **Test Accuracy**
- Most important for performance
- How well does it work on new data?
- Higher is better

#### 2. **Training Time**
- Practical consideration
- How long to iterate?
- Lower is better

#### 3. **Parameter Count**
- Model size
- Memory requirements
- Deployment considerations
- Lower is better (if accuracy same)

#### 4. **Overfitting Gap**
- Train accuracy - Validation accuracy
- Smaller gap = better generalization
- Want close together

### The Comparison Table

```
Architecture    Parameters   Train Acc   Val Acc   Test Acc    Time
-----------------------------------------------------------------
simple          123,456      92.50%      90.20%    89.80%      45.2s
balanced        456,789      95.30%      92.40%    91.90%      78.5s
deep            789,012      97.20%      91.80%    91.50%      145.3s
wide          1,234,567      98.50%      90.10%    89.20%      98.7s
residual        654,321      96.80%      93.50%    92.80%      112.4s
```

### How to Interpret Results

#### Scenario 1: Similar Accuracy
```
simple:    90% in 45s
balanced:  91% in 78s
```
**Choose:** Simple! 1% isn't worth 73% more time

#### Scenario 2: Big Accuracy Difference
```
simple:    85% in 45s
balanced:  92% in 78s
```
**Choose:** Balanced! 7% improvement is significant

#### Scenario 3: Overfitting
```
deep:  Train=98%, Val=88% (10% gap)
balanced: Train=94%, Val=92% (2% gap)
```
**Choose:** Balanced! Better generalization

#### Scenario 4: Production Deployment
```
wide:     92% with 2M parameters
residual: 93% with 600K parameters
```
**Choose:** Residual! Similar accuracy, 3x smaller

## Experiments to Try

### 1. **Test All Architectures**
```python
# Uncomment all in the list
architectures_to_test = [
    'simple',
    'balanced',
    'deep',
    'wide',
    'residual',
]
```

### 2. **Change Dataset**
```python
DATASET_CHOICE = 3  # Try CIFAR-10
```
**Question:** Which architecture wins on harder data?

### 3. **Create Your Own Architecture**
```python
elif architecture_name == 'my_custom':
    model = keras.Sequential([
        # Your design here!
        # Mix and match ideas from other architectures
    ])
```

### 4. **Modify Existing Architectures**

**Make Balanced Deeper:**
```python
# Add another Conv block
keras.layers.Conv2D(256, (3, 3), activation='relu'),
keras.layers.BatchNormalization(),
```

**Make Deep Wider:**
```python
# Increase filter numbers
Conv2D(64) → Conv2D(96)
Conv2D(128) → Conv2D(192)
```

### 5. **Hyperparameter Grid Search**

Test combinations:
```python
learning_rates = [0.001, 0.0001]
batch_sizes = [32, 64, 128]
dropout_rates = [0.3, 0.5, 0.7]

# Test all combinations
# Record which works best
```

### 6. **Different Optimizers**
```python
# Try different optimizers
optimizer='adam'                                    # Default
optimizer=keras.optimizers.SGD(momentum=0.9)       # Classic
optimizer=keras.optimizers.RMSprop()               # Alternative
optimizer=keras.optimizers.AdamW(weight_decay=0.01) # Modern
```

## Design Principles

### When to Go Deeper (More Layers)
✅ Complex datasets (CIFAR-10, real photos)
✅ Need hierarchical feature learning
✅ Have sufficient data (1000+ samples per class)
✅ Can afford training time

❌ Simple patterns (MNIST)
❌ Limited data
❌ Need fast inference

### When to Go Wider (More Filters)
✅ Rich, diverse features in data
✅ Large dataset
✅ High-resolution images
✅ Computational power available

❌ Limited data (will overfit)
❌ Simple patterns
❌ Memory constraints
❌ Mobile deployment

### When to Add Regularization
✅ Overfitting observed (train >> val)
✅ Limited data
✅ Complex model
✅ Better generalization needed

**Regularization techniques:**
- Dropout (0.3-0.7)
- Batch Normalization
- L1/L2 weight regularization
- Data augmentation

### When to Use Skip Connections
✅ Very deep networks (10+ layers)
✅ Vanishing gradient problems
✅ Need to preserve low-level features
✅ Research/advanced projects

❌ Simple, shallow networks
❌ First projects (adds complexity)

## Common Architecture Patterns

### VGG-style
```python
Conv(64) → Conv(64) → MaxPool
Conv(128) → Conv(128) → MaxPool
Conv(256) → Conv(256) → Conv(256) → MaxPool
# Progressive increase in filters
```

### ResNet-style
```python
x = Conv(64)(x)
residual = x
x = Conv(64)(x)
x = Conv(64)(x)
x = Add()([x, residual])
# Skip connections every 2-3 layers
```

### Inception-style
```python
# Multiple filter sizes in parallel
branch1 = Conv2D(64, (1,1))(x)
branch2 = Conv2D(64, (3,3))(x)
branch3 = Conv2D(64, (5,5))(x)
x = Concatenate()([branch1, branch2, branch3])
# Multi-scale feature extraction
```

## Debugging Poor Performance

### Accuracy not improving (<60%)
1. ✅ Check learning rate (try 0.001, 0.0001)
2. ✅ Ensure data is normalized
3. ✅ Check labels are correct
4. ✅ Try simpler architecture first
5. ✅ Increase epochs

### Overfitting (train 95%, val 75%)
1. ✅ Add dropout (0.5)
2. ✅ Add data augmentation
3. ✅ Reduce model size
4. ✅ Get more training data
5. ✅ Add L2 regularization

### Underfitting (both accuracies low)
1. ✅ Increase model capacity
2. ✅ Add more layers
3. ✅ Increase filter numbers
4. ✅ Train for more epochs
5. ✅ Decrease regularization

### Training too slow
1. ✅ Reduce model size
2. ✅ Increase batch size
3. ✅ Use fewer epochs for testing
4. ✅ Reduce image resolution
5. ✅ Use simpler architecture

## Key Takeaways

💡 **No single "best" architecture** - depends on your needs
💡 **Simple often wins** for simple problems
💡 **More parameters ≠ better** - risk of overfitting
💡 **Balance is key** - accuracy vs speed vs size
💡 **Experiment systematically** - change one thing at a time
💡 **Understand trade-offs** - every choice has consequences

## Your CNN Journey - Complete! 🎉

### What You've Mastered

**Project 1:** Basic CNN (Conv, Pool, Dense)
**Project 2:** Regularization (Dropout, BatchNorm)
**Project 3:** Color images (RGB, deeper networks)
**Project 4:** Data augmentation (transformations)
**Project 5:** Architecture design (systematic comparison)

### You Can Now:
✅ Build CNNs from scratch
✅ Understand each layer's purpose
✅ Prevent overfitting
✅ Handle different image types
✅ Design custom architectures
✅ Compare models systematically
✅ Tune hyperparameters
✅ Debug training issues

## Next Level Challenges

### 1. **Transfer Learning**
Use pre-trained models (VGG16, ResNet50, EfficientNet)
```python
base = keras.applications.ResNet50(weights='imagenet')
```

### 2. **Object Detection**
Find objects in images (YOLO, Faster R-CNN)

### 3. **Semantic Segmentation**
Pixel-level classification (U-Net, DeepLab)

### 4. **GANs**
Generate new images (Generative Adversarial Networks)

### 5. **Attention Mechanisms**
Focus on important parts (Transformers, Vision Transformers)

### 6. **Real-World Deployment**
- Flask/FastAPI web service
- TensorFlow Lite for mobile
- ONNX for cross-platform
- Docker containers

### 7. **Your Final Year Project!**
Apply everything you've learned to solve a real problem

## Questions to Ponder

1. Why does the "simple" architecture sometimes beat "deep" on easy datasets?
2. How do you decide between accuracy and inference speed?
3. What would you change if deploying to a smartphone?
4. How would you design a CNN for medical imaging?
5. What's the relationship between data size and model complexity?

## Final Advice

1. **Understand before optimizing** - Know why it works
2. **Start simple** - Add complexity only when needed
3. **Visualize everything** - See what the model learns
4. **Document experiments** - Track what works
5. **Share knowledge** - Teaching helps you learn
6. **Keep learning** - Field evolves rapidly
7. **Build projects** - Best way to solidify understanding

---

## 🎓 Congratulations!

You've completed the CNN Practice Projects series! You now have:
- Strong foundation in CNNs
- Practical coding skills
- Ability to design & debug models
- Understanding of key concepts
- Confidence for your final year

**You're ready for real-world projects!** 🚀

Remember: Every expert started where you are now. Keep experimenting, stay curious, and build amazing things!

---

**Questions? Ideas? Found bugs?**
Review your project files, experiment freely, and most importantly - HAVE FUN! 🎉
