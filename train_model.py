import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# === CONFIGURATION ===
MODEL_NAME = 'asl_mobilenetv2_fine_tuning_captured_data'
FREEZE_BACKBONE = False  # Set to True to freeze all layers except classifier
USE_PRETRAINED_MODEL = True  # Set to True to load your own pretrained model
PRETRAINED_MODEL_PATH = os.path.join('models', 'asl_mobilenetv2_kaggle', 'asl_mobilenetv2_kaggle.pth')
DATA_DIR = 'captured_train_data' #os.path.join('datasets', 'grassknoted', 'asl-alphabet', 'versions', '1', 'asl_alphabet_train', 'asl_alphabet_train') 
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_WORKERS = 4
LR = 0.0001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 29 

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)) # Apply blur with 50% chance
    ], p=0.5),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets and loaders
train_dataset = ImageFolder(DATA_DIR, transform=train_transform)
train_loader = DataLoader(train_dataset, BATCH_SIZE=BATCH_SIZE, shuffle=True, NUM_WORKERS=NUM_WORKERS)

# Model
if USE_PRETRAINED_MODEL and os.path.exists(PRETRAINED_MODEL_PATH):
    model = mobilenet_v2(num_classes=num_classes)
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
else:
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# Freeze backbone if specified
if FREEZE_BACKBONE:
    for param in model.features.parameters():
        param.requires_grad = False

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()) , LR=LR)

# Training loop
train_losses, train_accuracies = [], []

def main():
    print("Training on device:", device)
    if USE_PRETRAINED_MODEL:
        print(f"Using pretrained model from {PRETRAINED_MODEL_PATH}")
        
    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs", position=0):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        # tqdm for batches
        batch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}",position=1, leave=False)

        for images, labels in batch_iter:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss and accuracy
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            batch_iter.set_postfix(batch_loss=loss.item())

        # Calculate epoch loss and accuracy    
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        tqdm.write(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    print("Training complete.")

    # experiment directory
    experiment_path = os.path.join('models', MODEL_NAME)

    # Save model
    model_save_path = os.path.join(experiment_path, f'{MODEL_NAME}.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    # Plot training loss and accuracy
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_save_path = os.path.join(experiment_path, f'{MODEL_NAME}_train.png')
    plt.savefig(plot_save_path)

    print(f"Model and train loss/accuracies plot saved to {experiment_path}")

if __name__ == "__main__":
    main()
