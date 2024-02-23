import torch
import torch.nn as nn
import torch.optim as optim

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 32)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Create an instance of the MLP model
model = MLP()

# Generate fake input data
input_data = torch.randn(10, 32)  # 10 samples with 32 features

# Generate fake target labels
target_labels = torch.randint(0, 2, (10,))  # 10 binary labels

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Forward pass
    output = model(input_data)
    loss = criterion(output, target_labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss for every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# Test the trained model on new data
test_data = torch.randn(5, 32)  # 5 samples for testing
with torch.no_grad():
    model.eval()
    predictions = model(test_data)
    predicted_labels = torch.argmax(predictions, dim=1)
    print("Predicted Labels:", predicted_labels)