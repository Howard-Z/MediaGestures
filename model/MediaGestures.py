from HandLandMarkDataset import *
from train_val import *
import torch.nn as nn
import torch

NUM_LABELS = 2

class MediaGesture(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None

        self.fc1 = nn.Linear(63, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, NUM_LABELS)

        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('relu'))

        self.relu = nn.ReLU()
        

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    
#Define the model, optimizer, and criterion (loss_fn)
model = MediaGesture()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-4,)

criterion = nn.CrossEntropyLoss()


# Define the dataset and data transform with flatten functions appended

# tester = HandLandmarkDataset(os.path.join(get_root_dir(), "parsed_data"))

train_dataset = HandLandmarkDataset(os.path.join(get_root_dir(), "parsed_data"))

val_dataset = HandLandmarkDataset(os.path.join(get_root_dir(), "parsed_data"))

# Define the batch size and number of workers
batch_size = 1
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
print(train_dataset[69])
print(model(torch.from_numpy(train_dataset[69][0])))

