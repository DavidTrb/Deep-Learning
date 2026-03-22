import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.datasets import fetch_openml
import deustorch.nn, deustorch.models, deustorch.optim

# ---- 1. Load and preprocess data ----
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_all = mnist.data.astype(np.float32) / 255.0
y_all = mnist.target.astype(int)
X_train, X_test = X_all[:60000], X_all[60000:]
y_train, y_test = y_all[:60000], y_all[60000:]

def one_hot(labels, C=10):
    Y = np.zeros((len(labels), C), dtype=np.float32)
    Y[np.arange(len(labels)), labels] = 1.0
    return Y

Y_train, Y_test = one_hot(y_train), one_hot(y_test)

# ---- 2. Build model ----
model = deustorch.models.MLP4()

# ---- 3. Kaiming initialization ----
np.random.seed(42)
for layer in model.layers:
    fan_in = layer.W.shape[1]
    layer.W = np.random.randn(*layer.W.shape).astype(np.float32) * np.sqrt(2 / fan_in)
    layer.b = np.zeros(layer.b.shape, dtype=np.float32)

# ---- 4. Loss and optimizer ----
criterion = deustorch.nn.CrossEntropyLoss()
optimizer = deustorch.optim.SGD(model, lr=0.1, momentum=0.9)

# ---- 5. Mini-batch training loop ----
BATCH_SIZE, NUM_EPOCHS = 128, 20

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_loss = 0.0
    num_batches = 0
    idx = np.random.permutation(len(X_train))

    for start in range(0, len(X_train), BATCH_SIZE):
        b = idx[start : start + BATCH_SIZE]

        # Forward pass
        logits = model.forward(X_train[b])

        # Compute loss
        loss = criterion.forward(logits, Y_train[b])

        # Backward pass
        dLdA = criterion.backward()

        # Propagate gradients
        model.backward(dLdA)

        # Update weights
        optimizer.step()

        # Acumular loss
        epoch_loss += loss
        num_batches += 1

    # ---- 6. Evaluate accuracy ----
    train_preds = np.argmax(model.forward(X_train), axis=1)
    test_preds  = np.argmax(model.forward(X_test),  axis=1)

    train_acc = np.mean(train_preds == y_train)
    test_acc  = np.mean(test_preds  == y_test)

    # ---- Print stats ----
    print(f"Epoch {epoch:2d} | Loss: {epoch_loss/num_batches:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")