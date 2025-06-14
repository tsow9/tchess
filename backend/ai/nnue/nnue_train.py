from datetime import datetime
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from backend.ai.nnue.nnue_model import SimpleNNUE

from backend.ai.constants import local_paths
from backend.ai.utils.save_latest_graphs import save_latest_graphs_from_logs

# parameters
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset name to load
DATASET = "20250613_1605_AlphaBetaAI_d=2_vs_AlphaBetaAI_d=3"
GRAPH_TITLE = "Training_Loss" + DATASET
X_LABEL = "Epoch"
Y_LABEL = "Loss"

# load data
data = torch.load(os.path.join(local_paths.DATA_SELF_LEARNING_DATASET, f"{DATASET}.pt"))

print("Converting data to tensors...")
inputs = torch.stack([x[0] for x in data]).to(DEVICE)  # shape: (N, 768)
targets = torch.tensor([x[1] for x in data], dtype=torch.float32).unsqueeze(1).to(DEVICE)  # shape: (N, 1)

dataset = TensorDataset(inputs, targets)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = SimpleNNUE().to(DEVICE)
criterion = nn.MSELoss()   # Mean Square Error
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Training started...")

# Save training history
loss_history = []

for epoch in range(EPOCHS):
  total_loss = 0
  for x, y in loader:   # Loop by BATCH_SIZE
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

    avg_loss = total_loss / len(loader)   # = BATCH_SIZE
  loss_history.append(avg_loss)

  print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# save logs
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
log_path = os.path.join(local_paths.DATA_TRAINING_LOGS, f"log_{timestamp}.json")
with open(log_path, "w") as f:
  json.dump(loss_history, f)

# save graphs
save_latest_graphs_from_logs(
  log_dir=local_paths.DATA_TRAINING_LOGS, out_dir=local_paths.DATA_TRAINING_GRAPHS, 
  title=GRAPH_TITLE, x_label=X_LABEL, y_label=Y_LABEL, max_files=5
)

# save model
torch.save(model.state_dict(), os.path.join(local_paths.DATA_MODELS, f"model_D{DATASET}_E{EPOCHS}_B{BATCH_SIZE}.pth"))
print(f"Training completed. Model saved to model_D{DATASET}_E{EPOCHS}_B{BATCH_SIZE}.pth")


