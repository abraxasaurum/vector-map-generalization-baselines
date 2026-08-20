from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEQUENCE_FILE = PROJECT_ROOT / "data" / "processed" / "building_sequences_32points.npz"
MODEL_FILE = PROJECT_ROOT / "outputs" / "cnn_baseline.pt"
HISTORY_FILE = PROJECT_ROOT / "outputs" / "cnn_training_history.csv"
FIGURE_FILE = PROJECT_ROOT / "outputs" / "cnn_test_predictions.png"

SEED = 42
TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.15
BATCH_SIZE = 32
MAX_EPOCHS = 400
PATIENCE = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

data = np.load(SEQUENCE_FILE)
X = data["X"].astype(np.float32)
y = data["y"].astype(np.float32)

n_buildings, n_points, n_coordinates = X.shape

indices = np.random.permutation(n_buildings)

n_test = int(np.ceil(n_buildings * TEST_FRACTION))
test_indices = indices[:n_test]

remaining_indices = indices[n_test:]
n_validation = int(np.ceil(len(remaining_indices) * VALIDATION_FRACTION))

validation_indices = remaining_indices[:n_validation]
train_indices = remaining_indices[n_validation:]

X_train = torch.from_numpy(X[train_indices])
y_train = torch.from_numpy(y[train_indices])

X_validation = torch.from_numpy(X[validation_indices])
y_validation = torch.from_numpy(y[validation_indices])

X_test = torch.from_numpy(X[test_indices])
y_test = torch.from_numpy(y[test_indices])

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
)


class ResidualCircularCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=2, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(16, 2, kernel_size=5, padding=2, padding_mode="circular"),
        )

        # Das Modell startet exakt als Identity-Modell.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x):
        x_channels_first = x.transpose(1, 2)
        correction = self.network(x_channels_first).transpose(1, 2)
        return x + correction


model = ResidualCircularCNN().to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
loss_function = nn.MSELoss()

n_parameters = sum(parameter.numel() for parameter in model.parameters())
identity_validation_mse = torch.mean((X_validation - y_validation) ** 2).item()
identity_test_mse = torch.mean((X_test - y_test) ** 2).item()

print(
    f"Buildings: {n_buildings} | "
    f"Train: {len(train_indices)} | "
    f"Validation: {len(validation_indices)} | "
    f"Test: {len(test_indices)}"
)
print(f"Trainable parameters: {n_parameters}")
print(f"Identity validation MSE: {identity_validation_mse:.6f}")
print(f"Identity test MSE: {identity_test_mse:.6f}")

best_validation_mse = identity_validation_mse
best_epoch = 0
best_state = deepcopy(model.state_dict())
epochs_without_improvement = 0
history = []

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    train_losses = []

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = loss_function(predictions, batch_y)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        validation_predictions = model(X_validation.to(device))
        validation_mse = loss_function(
            validation_predictions,
            y_validation.to(device),
        ).item()

    train_mse = float(np.mean(train_losses))
    history.append(
        {
            "epoch": epoch,
            "train_mse": train_mse,
            "validation_mse": validation_mse,
        }
    )

    if validation_mse < best_validation_mse - 1e-8:
        best_validation_mse = validation_mse
        best_epoch = epoch
        best_state = deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 50 == 0:
        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"train MSE: {train_mse:.6f} | "
            f"validation MSE: {validation_mse:.6f}"
        )

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch}.")
        break

model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    test_predictions = model(X_test.to(device))
    final_test_mse = loss_function(test_predictions, y_test.to(device)).item()

history_df = pd.DataFrame(history)
history_df.to_csv(HISTORY_FILE, index=False)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "n_points": n_points,
        "n_coordinates": n_coordinates,
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation_mse,
    },
    MODEL_FILE,
)

improvement = 1 - final_test_mse / identity_test_mse

print(f"\nBest epoch: {best_epoch}")
print(f"Best validation MSE: {best_validation_mse:.6f}")
print(f"Final test MSE: {final_test_mse:.6f}")
print(f"Improvement over identity baseline: {improvement:.1%}")

source_test = X_test.numpy()
target_test = y_test.numpy()
predicted_test = test_predictions.cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

for axis, example_index in zip(axes, range(3)):
    source_ring = np.vstack([source_test[example_index], source_test[example_index][0]])
    target_ring = np.vstack([target_test[example_index], target_test[example_index][0]])
    predicted_ring = np.vstack([predicted_test[example_index], predicted_test[example_index][0]])

    axis.plot(
        source_ring[:, 0],
        source_ring[:, 1],
        "--",
        color="#4C78A8",
        linewidth=1.2,
        label="Source",
    )
    axis.plot(
        target_ring[:, 0],
        target_ring[:, 1],
        color="#E45756",
        linewidth=2,
        label="Target",
    )
    axis.plot(
        predicted_ring[:, 0],
        predicted_ring[:, 1],
        color="#54A24B",
        linewidth=1.4,
        label="1D-CNN prediction",
    )

    axis.set_title(f"Test example {example_index + 1}")
    axis.set_aspect("equal")
    axis.grid(alpha=0.25)

axes[0].legend(loc="best")
fig.suptitle("Residual circular 1D-CNN: source, target and prediction", y=1.02)
plt.tight_layout()
plt.savefig(FIGURE_FILE, dpi=180, bbox_inches="tight")
plt.close()

print(f"\nModel: {MODEL_FILE}")
print(f"Training history: {HISTORY_FILE}")
print(f"Prediction figure: {FIGURE_FILE}")
