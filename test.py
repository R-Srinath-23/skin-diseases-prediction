import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import torch.nn as nn

# Dataset directory
data_dir = r"C:\ML Project\Skin_disease_dataset\Mendeley_Data"

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mobilenet_v2 = models.mobilenet_v2(weights=None)

num_classes = len(dataset.classes)


# Load model and modify classifier
mobilenet_v2.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1280, 128),
    nn.ReLU(),
    nn.Linear(128, num_classes)
)
mobilenet_v2.to(device)

# Load trained weights
mobilenet_v2.load_state_dict(torch.load(r"C:\ML Project\skin_prediction\skin_disease_mobilenetv2_1weights.pth"))
mobilenet_v2.eval()

# Evaluation
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = mobilenet_v2(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
