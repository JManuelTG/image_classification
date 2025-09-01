# model.py
import torch
from torchvision import models, transforms
from PIL import Image
import json

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    classification_model = models.resnet50(pretrained=True).to(device)
    classification_model.eval()

    feature_extractor = torch.nn.Sequential(
        *list(classification_model.children())[:-1]
    ).to(device)
    feature_extractor.eval()

    with open("data/imagenet.json", "r") as f:
        classes = json.load(f)

    return classification_model, feature_extractor, device, classes


def preprocess_image(file, device):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    image = Image.open(file.stream).convert("RGB")
    return preprocess(image).unsqueeze(0).to(device)


def predict_class(model, tensor, classes):
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = outputs.max(1)
        return classes[predicted.item()]


def extract_features(feature_extractor, tensor):
    with torch.no_grad():
        features = feature_extractor(tensor)
        return torch.flatten(features, start_dim=1).squeeze().cpu()
