import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 32
learning_rate = 0.01  # Adjusted for SGD
num_epochs = 15
weight_decay = 1e-4  # L2 Regularization
dropout_rate = 0.3  # Dropout to prevent overfitting

# Data Augmentation & Normalization
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dataset path
data_dir = "images"
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

# Compute class weights for imbalanced dataset
labels = [label for _, label in train_dataset.samples]
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# CNN Model with Dropout
class OralHistologyCNN(nn.Module):
    def __init__(self):
        super(OralHistologyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Initialize model, loss, optimizer, and scheduler
    model = OralHistologyCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Weighted loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)  # SGD + Momentum
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce LR after 5 epochs

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()  # Adjust learning rate
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    print("Training Complete!")

    # Save model correctly
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_weights': class_weights,
        'hyperparameters': {
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'dropout_rate': dropout_rate
        }
    }, "oral_histology_cnn.pth")
    print("Model Saved!")

    # Evaluation Function
    def evaluate_model(model, dataloader, dataset_name):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device).long()
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="binary")
        recall = recall_score(all_labels, all_preds, average="binary")
        f1 = f1_score(all_labels, all_preds, average="binary")
        cm = confusion_matrix(all_labels, all_preds)
        print(f"\n{dataset_name} Set Performance:")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        return accuracy, precision, recall, f1, cm

    # Load model for evaluation
    checkpoint = torch.load("oral_histology_cnn.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Evaluate on Validation and Test Set
    evaluate_model(model, val_loader, "Validation")
    evaluate_model(model, test_loader, "Test")
