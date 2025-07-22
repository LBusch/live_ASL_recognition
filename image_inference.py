import os
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import csv

# === CONFIGURATION ===
MODEL_NAME = 'asl_mobilenetv2_fine_tuning_captured_data'
IMAGE_DIR = 'captured_test_data\\A' # os.path.join('datasets', 'grassknoted', 'asl-alphabet', 'versions', '1', 'asl_alphabet_test', 'asl_alphabet_test')

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join('models', MODEL_NAME, f'{MODEL_NAME}.pth')
model = mobilenet_v2(num_classes=29)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load labels from example directory
asl_example_dir = 'example_data'
labels = sorted([
    d for d in os.listdir(asl_example_dir) 
    if os.path.isdir(os.path.join(asl_example_dir, d))
])

# Function to infer a single image
def infer_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        pred = outputs.argmax(dim=1).item()
        return labels[pred]

# Iterate through images in the directory and infer
results = []
for fname in os.listdir(IMAGE_DIR):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(IMAGE_DIR, fname)
        pred_class = infer_image(img_path)
        print(f"{fname}: {pred_class}")
        results.append({'filename': fname, 'predicted_label': pred_class})

# Save results as CSV in inference directory
inference_dir = os.path.join('models', MODEL_NAME, 'inference_results')
os.makedirs(inference_dir, exist_ok=True)
csv_path = os.path.join(inference_dir, 'inference_results.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['filename', 'predicted_label'], delimiter=';')
    writer.writeheader()
    writer.writerows(results)

print(f"Inference results saved to {csv_path}")

