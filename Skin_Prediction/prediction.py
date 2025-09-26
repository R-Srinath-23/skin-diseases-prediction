import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import Image
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights
import torch.nn.functional as F

PREDICTION_MODEL=r"C:\ML Project\skin_prediction\skin_disease_mobilenetv2_weights.pth"
data_dir = r"C:\ML Project\Skin_disease_dataset\Mendeley_Data"
INPUT_IMAGE="C:\ML Project\input_img\images.jpg"

# datasets.ImageFolder(data_dir, transform=transform).classes
DATA_SET_CLASSES = ['Nail_psoriasis', 'SJS-TEN', 'Vitiligo', 'acne', 'hyperpigmentation']

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # adjust to match training
])

# Load the full dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Random split
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


# batch_size = 32
# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# weights=MobileNet_V2_Weights.DEFAULT

# class_names = dataset.classes

# mobilenet_v2 = models.mobilenet_v2(pretrained=True)
# mobilenet_v2 = models.mobilenet_v2(weights='DEFAULT')

# Replace the classifier of the model
print("Starting...")
mobilenet_v2 = models.mobilenet_v2(weights='DEFAULT')
mobilenet_v2.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1280, 128),
    nn.ReLU(),
    nn.Linear(128, out_features=5)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loaded model & device...")

mobilenet_v2 = mobilenet_v2.to(device)
mobilenet_v2.load_state_dict(torch.load(PREDICTION_MODEL))
mobilenet_v2.eval()  # Set to evaluation mode

def predict_image(image):
    mobilenet_v2.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = mobilenet_v2(img)
        probs = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

    predicted_label = DATA_SET_CLASSES[predicted_class.item()]
    return predicted_label


if __name__ == "__main__":
    # print(predict_image(INPUT_IMAGE, mobilenet_v2, train_dataset.dataset.classes))
    print(predict_image(INPUT_IMAGE))