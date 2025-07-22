import os
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from collections import defaultdict
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

# === CONFIGURATION ===
MODEL_NAME = 'asl_mobilenetv2_fine_tuning_captured_data'
BATCH_SIZE = 32
NUM_WORKERS = 4
TEST_DATA_DIR = 'captured_test_data'

# Experiment directory
experiment_dir = os.path.join('models', MODEL_NAME)

# Image transformations
test_transforms = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test dataset
test_dataset = ImageFolder(TEST_DATA_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(experiment_dir, f'{MODEL_NAME}.pth')
model = mobilenet_v2(num_classes=len(test_dataset.classes))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Evaluation
correct = 0
total = 0
class_correct = defaultdict(int)
class_total = defaultdict(int)

all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Test Batch", position=0):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Update class-wise accuracy
        for label, pred in zip(labels, preds):
            class_total[label.item()] += 1
            if label == pred:
                class_correct[label.item()] += 1

        # Collect for confusion matrix
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Calculate accuracy
overall_accuracy = correct / total

# Calculate class-wise accuracy
class_accuracies = {
    test_dataset.classes[i]: class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
    for i in range(len(test_dataset.classes))
}

# Calculate confusion matrix
cm = confusion_matrix(all_labels, all_preds)
cm_list = cm.tolist()  # For JSON serialization

# Save confusion matrix plot
test_dir = os.path.join(experiment_dir, 'test_results')
os.makedirs(test_dir, exist_ok=True)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues', colorbar=False)
plt.tight_layout()
plt.savefig(os.path.join(test_dir, 'confusion_matrix.png'))
plt.close(fig)

# Save results as CSV
csv_path = os.path.join(test_dir, 'test_accuracies.csv')
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    # Header for class-wise accuracy
    writer.writerow(['class', 'accuracy'])
    for cls, acc in class_accuracies.items():
        writer.writerow([cls, acc])
    # Last row: overall accuracy
    writer.writerow(['all classes', overall_accuracy])

# Print results
print(f"Overall accuracy: {overall_accuracy:.4f}")
for cls, acc in class_accuracies.items():
    print(f"Accuracy for class '{cls}': {acc:.4f}")
print(f"Confusion matrix plot saved to {os.path.join(test_dir, 'confusion_matrix.png')}")
print(f"Accuracies saved to {csv_path}")