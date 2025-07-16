import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import timm
import numpy as np
from collections import Counter

# --- Configuration ---
# Device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == 'cpu':
    print("Error: This script is designed to run on a GPU. Please ensure CUDA is available.")
    exit()

# Dataset path
data_dir = "ham10000_processed"

# Model choice - More robust models are better for real-world generalization
# Options: "efficientnet_b3", "convnext_tiny", "efficientnet_b0", "mobilenet_v3"
MODEL_CHOICE = "efficientnet_b3"

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 30 # Increased epochs for more robust training
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
# --- End Configuration ---


# --- Data Augmentation for Real-World Generalization ---
# These transforms are more aggressive to simulate real-world image variations
train_transform = transforms.Compose([
    transforms.Resize((256, 256)), # Start with a slightly larger size before cropping
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(60), # Increased rotation
    transforms.RandomPerspective(distortion_scale=0.3, p=0.3), # Added perspective transform
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2), # Stronger color jitter
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)), # Added blur
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- Focal Loss (for class imbalance) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# --- Main Training Script ---
if __name__ == "__main__":
    # --- Dataset Loading and Splitting ---
    full_dataset = datasets.ImageFolder(root=data_dir)
    num_classes = len(full_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {full_dataset.classes}")
    
    # Train/val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(range(len(full_dataset)), [train_size, val_size])
    
    # Create separate datasets with the correct transforms
    train_dataset = torch.utils.data.Subset(datasets.ImageFolder(root=data_dir, transform=train_transform), train_indices.indices)
    val_dataset = torch.utils.data.Subset(datasets.ImageFolder(root=data_dir, transform=val_transform), val_indices.indices)

    # --- WeightedRandomSampler for Class Imbalance ---
    print("\nHandling class imbalance with WeightedRandomSampler...")
    # Get labels for the training split
    train_labels = [full_dataset.targets[i] for i in train_indices.indices]
    
    # Count class occurrences
    class_counts = Counter(train_labels)
    print(f"Training class distribution: {class_counts}")
    
    # Calculate weights for each class (inverse of frequency)
    class_weights = {class_id: 1.0 / count for class_id, count in class_counts.items()}
    
    # Create a weight for each sample in the training set
    sample_weights = [class_weights[label] for label in train_labels]
    
    # Create the sampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    print("[SUCCESS] WeightedRandomSampler created.")

    # --- DataLoaders ---
    # Note: shuffle=False because the sampler handles random sampling.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model Selection ---
    print(f"\nLoading model: {MODEL_CHOICE}...")
    model = timm.create_model(MODEL_CHOICE, pretrained=True, num_classes=num_classes)
    model_name = MODEL_CHOICE
    
    # Add dropout to the classifier for regularization
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        dropout_rate = 0.4 # Slightly increased dropout
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            model.classifier
        )
        print(f"Added Dropout ({dropout_rate}) to the classifier.")

    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # --- Training Setup ---
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()

    # --- Training Loop ---
    train_accs, val_accs, train_losses, val_losses = [], [], [], []
    best_val_acc = 0.0

    print(f"\nStarting training with {model_name} for {NUM_EPOCHS} epochs...")
    print("-" * 80)

    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx > 0 and batch_idx % 25 == 0:
                print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        train_acc = correct / total
        train_loss /= len(train_loader)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        val_loss /= len(val_loader)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{model_name}_realworld.pth")
            print(f"[SUCCESS] New best model saved with val_acc: {val_acc:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Current LR: {current_lr:.6f}")
        print("-" * 80)

    # --- Finalization and Plotting ---
    torch.save(model.state_dict(), f"final_{model_name}_realworld.pth")
    print(f"[SUCCESS] Final model saved as final_{model_name}_realworld.pth")
    print(f"[SUCCESS] Best validation accuracy achieved: {best_val_acc:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(range(1, NUM_EPOCHS+1), train_accs, label='Train Accuracy', marker='o')
    ax1.plot(range(1, NUM_EPOCHS+1), val_accs, label='Validation Accuracy', marker='s')
    ax1.axhline(y=best_val_acc, color='r', linestyle='--', label=f'Best Val Acc: {best_val_acc:.4f}')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"{model_name.upper()} - Training and Validation Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.5)
    
    ax2.plot(range(1, NUM_EPOCHS+1), train_losses, label='Train Loss', marker='o')
    ax2.plot(range(1, NUM_EPOCHS+1), val_losses, label='Validation Loss', marker='s')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"{model_name.upper()} - Training and Validation Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_realworld_training_plot.png", dpi=300)
    plt.show()

    print(f"\n[SUCCESS] Training completed!")
    print(f"[RESULTS] Final Results:")
    print(f"   - Model: {model_name}")
    print(f"   - Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"   - Total Parameters: {total_params:,}")
    print(f"   - Training completed in {NUM_EPOCHS} epochs")
