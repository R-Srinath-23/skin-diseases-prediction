import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from tqdm import tqdm

# Dataset directory
data_dir = r"C:\ML Project\Skin_disease_dataset\Mendeley_Data"


# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # adjust to match training
])

# Load the full dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Split sizes
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Random split
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

mobilenet_v2 = models.mobilenet_v2(pretrained=True)

for param in mobilenet_v2.features.parameters():
    param.requires_grad = False

num_classes = len(dataset.classes)
print(num_classes)

# Replace the classifier of the model
mobilenet_v2.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1280, 128),
    nn.ReLU(),
    nn.Linear(128, num_classes)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mobilenet_v2.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mobilenet_v2 = mobilenet_v2.to(device)


epochs = 5
for epoch in range(epochs):
    mobilenet_v2.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = mobilenet_v2(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%")

    torch.save(mobilenet_v2.state_dict(), "skin_disease_mobilenetv2_weights.pth")