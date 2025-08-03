from core import Value
from net_struc import MLP

# Training data
X = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
Y = [1.0, -1.0, -1.0, 1.0]

# Initialize model: 3 input neurons, two hidden layers of 4 neurons each, 1 output
mlp = MLP(3, [4, 4, 1])

# Training loop
learning_rate = 0.01
epochs = 20

for epoch in range(epochs):
    # Forward pass: compute predictions
    y_preds = [mlp(x) for x in X]

    # Compute loss (mean squared error)
    loss = sum((y_out - y_gt)**2 for y_out, y_gt in zip(y_preds, Y))

    # Backward pass
    for p in mlp.parameters():
        p.grad = 0.0
    loss.backward_pass()

    # Gradient descent step
    for p in mlp.parameters():
        p.data -= learning_rate * p.grad

    # Logging
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.data:.4f}")

# Final predictions
final_preds = [mlp(x).data for x in X]
print("\nFinal predictions:", final_preds)