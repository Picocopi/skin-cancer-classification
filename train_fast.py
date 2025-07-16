import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import timm  # For state-of-the-art models

# Device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset path
data_dir = "ham10000_processed"

# Simplified but effective transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    # Load dataset
    full_dataset = datasets.ImageFolder(root=data_dir)
    num_classes = len(full_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {full_dataset.classes}")
    
    # Train/val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(range(len(full_dataset)), [train_size, val_size])
    
    # Create separate datasets with different transforms
    train_dataset = torch.utils.data.Subset(datasets.ImageFolder(root=data_dir, transform=train_transform), train_indices.indices)
    val_dataset = torch.utils.data.Subset(datasets.ImageFolder(root=data_dir, transform=val_transform), val_indices.indices)

    # Data loaders - Optimized for speed
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # Choose a faster but better model than ResNet18
    model_choice = "efficientnet_b0"  # Fast and efficient
    
    if model_choice == "efficientnet_b0":
        # EfficientNet-B0 - Great balance of speed and accuracy
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        model_name = "efficientnet_b0"
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Simple but effective training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # Training parameters - Fast training
    num_epochs = 15  # Reduced for speed
    train_accs, val_accs, train_losses, val_losses = [], [], [], []
    best_val_acc = 0.0

    print(f"Starting FAST training with {model_name} for {num_epochs} epochs...")
    print("This should be faster than your previous models but still more accurate!")
    print("-" * 80)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print progress every 30 batches
            if batch_idx % 30 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

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
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{model_name}_fast.pth")
            print(f"[SUCCESS] New best model saved with val_acc: {val_acc:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Current LR: {current_lr:.6f}")
        print("-" * 80)

    # Save the final model
    torch.save(model.state_dict(), f"final_{model_name}_fast.pth")
    print(f"[SUCCESS] Final model saved as final_{model_name}_fast.pth")
    print(f"[SUCCESS] Best validation accuracy achieved: {best_val_acc:.4f}")

    # Enhanced plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(range(1, num_epochs+1), train_accs, label='Train Accuracy', marker='o', alpha=0.7)
    ax1.plot(range(1, num_epochs+1), val_accs, label='Validation Accuracy', marker='s', alpha=0.7)
    ax1.axhline(y=best_val_acc, color='r', linestyle='--', alpha=0.7, label=f'Best Val Acc: {best_val_acc:.4f}')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"{model_name.upper()} FAST - Training and Validation Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(range(1, num_epochs+1), train_losses, label='Train Loss', marker='o', alpha=0.7)
    ax2.plot(range(1, num_epochs+1), val_losses, label='Validation Loss', marker='s', alpha=0.7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"{model_name.upper()} FAST - Training and Validation Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_fast_training_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n[SUCCESS] FAST Training completed!")
    print(f"[RESULTS] Final Results:")
    print(f"   - Model: {model_name}")
    print(f"   - Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"   - Total Parameters: {total_params:,}")
    print(f"   - Training completed in {num_epochs} epochs")
    print(f"   - This model should be better than ResNet18 but faster than heavy models!")
