# 🧠 MNIST Neural Network from Scratch

This repository contains a ground-up implementation of a two-layer feedforward neural network designed for handwritten digit classification. Built using only **NumPy**, this project serves as a deep dive into the fundamental linear algebra and calculus that power modern deep learning, moving beyond the abstraction of high-level frameworks.

### 🔬 Technical Architecture
The network follows a classic MLP (Multi-Layer Perceptron) architecture with the following specifications:
*   **Input Layer ($a^{[0]}$):** 784 neurons (28x28 flattened grayscale pixels).
*   **Hidden Layer ($a^{[1]}$):** 10 neurons with **ReLU** (Rectified Linear Unit) activation.
*   **Output Layer ($a^{[2]}$):** 10 neurons with **Softmax** activation for multi-class probability distribution.

---

### 📐 Mathematical Framework

#### 1. Forward Propagation
The forward pass computes the transformations through each layer to generate a prediction.

$$Z^{[1]} = W^{[1]}X + b^{[1]}$$
$$A^{[1]} = g_{ReLU}(Z^{[1]})$$
$$Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}$$
$$A^{[2]} = \sigma_{Softmax}(Z^{[2]})$$



#### 2. Activation Functions
*   **ReLU:** Introduces non-linearity by thresholding values at zero: $f(x) = \max(0, x)$. This allows the model to learn complex patterns without the vanishing gradient issues common in sigmoid functions.
*   **Softmax:** Normalizes the output vector into a probability distribution where the sum of all elements equals 1, representing the network's confidence for each digit class.

#### 3. Backward Propagation
To minimize the cost function, we calculate the gradient of the loss with respect to every parameter using the chain rule (Backpropagation).

*   **Error at Output Layer:** $dZ^{[2]} = A^{[2]} - Y$
*   **Weight/Bias Gradients (Layer 2):** 
    $$dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}$$
    $$db^{[2]} = \frac{1}{m} \sum dZ^{[2]}$$
*   **Error at Hidden Layer:** $dZ^{[1]} = W^{[2]T} dZ^{[2]} \cdot g'_{ReLU}(Z^{[1]})$
*   **Weight/Bias Gradients (Layer 1):**
    $$dW^{[1]} = \frac{1}{m} dZ^{[1]} A^{[0]T}$$
    $$db^{[1]} = \frac{1}{m} \sum dZ^{[1]}$$

#### 4. Parameter Updates
Using Gradient Descent, we update the weights and biases in the direction that reduces the loss, scaled by the learning rate ($\alpha$).
$$W := W - \alpha dW$$
$$b := b - \alpha db$$

---

### 🚀 Implementation Highlights
*   **Vectorization:** Utilizing NumPy's dot products to process the entire training set ($m$ examples) simultaneously, significantly increasing computational efficiency compared to iterative loops.
*   **One-Hot Encoding:** Converting integer labels into binary vectors to facilitate the calculation of the error gradient ($A^{[2]} - Y$).
*   **Feature Scaling:** Standardizing pixel values from $[0, 255]$ to $[0, 1]$ to ensure stable weight updates and faster convergence during training.

### 📈 Results
The model achieves an accuracy of **~85%** on both training and validation (dev) sets. Given the simplicity of a single hidden layer with only 10 units, this demonstrates the power of the underlying mathematics in capturing the hierarchical features of numerical data.

### 🤝 Acknowledgments
This implementation was inspired by the instructional work of **Samson Zhang**. It serves as a pedagogical bridge between theoretical machine learning and practical code implementation.
