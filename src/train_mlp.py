from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEQUENCE_FILE = PROJECT_ROOT / "data" / "processed" / "building_sequences_32points.npz"
MODEL_FILE = PROJECT_ROOT / "outputs" / "mlp_baseline.pt"
HISTORY_FILE = PROJECT_ROOT / "outputs" / "mlp_training_history.csv"
FIGURE_FILE = PROJECT_ROOT / "outputs" / "mlp_test_predictions.png"

SEED = 42
TEST_FRACTION = 0.20
BATCH_SIZE = 32
EPOCHS = 600
LEARNING_RATE = 1e-3

np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

data = np.load(SEQUENCE_FILE)
X = data["X"].astype(np.float32)
y = data["y"].astype(np.float32)

n_buildings, n_points, n_coordinates = X.shape
input_size = n_points * n_coordinates

indices = np.random.permutation(n_buildings)
n_test = int(np.ceil(n_buildings * TEST_FRACTION))
test_indices = indices[:n_test]
train_indices = indices[n_test:]

X_train = torch.from_numpy(X[train_indices].reshape(-1, input_size))
y_train = torch.from_numpy(y[train_indices].reshape(-1, input_size))
X_test = torch.from_numpy(X[test_indices].reshape(-1, input_size))
y_test = torch.from_numpy(y[test_indices].reshape(-1, input_size))

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

class ResidualMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
        )

    def forward(self, x):
        return x + self.network(x)

model = ResidualMLP(input_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.MSELoss()

identity_mse = torch.mean((X_test - y_test) ** 2).item()
print(f"Buildings: {n_buildings} | Train: {len(train_indices)} | Test: {len(test_indices)}")
print(f"Identity baseline test MSE: {identity_mse:.6f}")

history = []

for epoch in range(1, EPOCHS + 1):
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
        test_predictions = model(X_test.to(device))
        test_loss = loss_function(test_predictions, y_test.to(device)).item()

    train_loss = float(np.mean(train_losses))
    history.append(
        {
            "epoch": epoch,
            "train_mse": train_loss,
            "test_mse": test_loss,
        }
    )

    if epoch == 1 or epoch % 100 == 0:
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train MSE: {train_loss:.6f} | "
            f"test MSE: {test_loss:.6f}"
        )

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "n_points": n_points,
        "n_coordinates": n_coordinates,
        "seed": SEED,
    },
    MODEL_FILE,
)

history_df = pd.DataFrame(history)
history_df.to_csv(HISTORY_FILE, index=False)

model.eval()
with torch.no_grad():
    predicted_test = model(X_test.to(device)).cpu().numpy()

source_test = X_test.numpy().reshape(-1, n_points, n_coordinates)
target_test = y_test.numpy().reshape(-1, n_points, n_coordinates)
predicted_test = predicted_test.reshape(-1, n_points, n_coordinates)

final_test_mse = history_df["test_mse"].iloc[-1]
improvement = 1 - final_test_mse / identity_mse

print(f"\nFinal test MSE: {final_test_mse:.6f}")
print(f"Improvement over identity baseline: {improvement:.1%}")

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
        linewidth=2.0,
        label="Target",
    )
    axis.plot(
        predicted_ring[:, 0],
        predicted_ring[:, 1],
        color="#54A24B",
        linewidth=1.4,
        label="MLP prediction",
    )

    axis.set_title(f"Test example {example_index + 1}")
    axis.set_aspect("equal")
    axis.grid(alpha=0.25)

axes[0].legend(loc="best")
fig.suptitle("Residual MLP: source, Douglas-Peucker target and prediction", y=1.02)
plt.tight_layout()
plt.savefig(FIGURE_FILE, dpi=180, bbox_inches="tight")
plt.close()

print(f"\nModel: {MODEL_FILE}")
print(f"Training history: {HISTORY_FILE}")
print(f"Prediction figure: {FIGURE_FILE}")
