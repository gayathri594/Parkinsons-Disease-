# import torch
# from torchvision import transforms
# from PIL import Image
# from model import get_model

# # # Load model
# # model = get_model()
# # model.eval()
# # Load model
# model = get_model()
# model.load_state_dict(torch.load("vgg_trained_weights.pt", map_location=torch.device("cpu")))  # if not using GPU
# model.eval()


# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# def predict_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     input_tensor = transform(image).unsqueeze(0)
#     output = model(input_tensor)
#     _, predicted = torch.max(output, 1)
#     # return "Parkinson" if predicted.item() == 1 else "Healthy"
#     return "Parkinson" if predicted.item() == 1 else "Healthy"

import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import os

# Load model
model = get_model()

# ✅ Safely get path to the trained weights
MODEL_PATH = os.path.join(os.path.dirname(__file__), "vgg_trained_weights.pt")
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    return "Parkinson" if predicted.item() == 1 else "Healthy"
