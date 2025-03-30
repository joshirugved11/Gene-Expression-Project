import torch
import torch.nn as nn
import torch.optim as optim

# Define training parameters
LEARNING_RATE = 0.001

class DLModel(nn.Module):
    def __init__(self, input_size):
        super(DLModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        # Removed explicit sigmoid here (Handled in loss function)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No sigmoid here
        return x

def train_dl_model(X, y, epochs=20, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32).view(-1, 1).to(device)

    model = DLModel(input_size=X.shape[1]).to(device)
    
    # Use BCEWithLogitsLoss (better for numerical stability)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model
