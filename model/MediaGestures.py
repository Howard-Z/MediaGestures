from HandLandMarkDataset import *
from train_val import *
import torch.nn as nn
import torch

class MediaGesture(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None

        self.fc1 = nn.Linear(63, 63)
        self.fc2 = nn.Linear(63, 63)
        self.fc3 = nn.Linear(63, 6)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    

batch_size = 64
num_workers = 2


#Define the model, optimizer, and criterion (loss_fn)
model = MediaGesture()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,)

criterion = nn.CrossEntropyLoss()


# Define the dataset and data transform with flatten functions appended

# tester = HandLandmarkDataset(os.path.join(get_root_dir(), "parsed_data"))

train_dataset = HandLandmarkDataset(os.path.join(get_root_dir(), "parsed_data"))

val_dataset = HandLandmarkDataset(os.path.join(get_root_dir(), "parsed_data"))

# Define the batch size and number of workers
batch_size = 32
num_workers = 0

# Define the data loaders

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=2)

print(train_dataset.len)