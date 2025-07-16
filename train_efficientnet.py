import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import timm  # For EfficientNet models

# Device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset path
data_dir = "ham10000_processed"

# Enhanced image transforms optimized for EfficientNet
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Choose EfficientNet model
    model_choice = "efficientnet_b3"  # Options: "efficientnet_b0", "efficientnet_b3", "efficientnet_b5"
    
    if model_choice == "efficientnet_b0":
        # EfficientNet-B0 - Fastest, ~5M parameters
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        model_name = "efficientnet_b0"
        print("Using EfficientNet-B0 - Fast and efficient (~5M parameters)")
        
    elif model_choice == "efficientnet_b3":
        # EfficientNet-B3 - Good balance, ~12M parameters
        model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes)
        model_name = "efficientnet_b3"
        print("Using EfficientNet-B3 - Excellent accuracy/speed balance (~12M parameters)")
        
    elif model_choice == "efficientnet_b5":
        # EfficientNet-B5 - Higher accuracy, ~30M parameters
        model = timm.create_model('efficientnet_b5', pretrained=True, num_classes=num_classes)
        model_name = "efficientnet_b5"
        print("Using EfficientNet-B5 - High accuracy (~30M parameters)")
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Training setup optimized for EfficientNet
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

    # Training parameters
    num_epochs = 15
    train_accs, val_accs, train_losses, val_losses = [], [], [], []
    best_val_acc = 0.0

    print(f"Starting {model_name.upper()} training for {num_epochs} epochs...")
    print("EfficientNet is known for excellent accuracy with good efficiency!")
    print("-" * 70)

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
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{model_name}_skin_cancer.pth")
            print(f"[BEST] New best model saved with val_acc: {val_acc:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
        print("-" * 70)

    # Save the final model
    torch.save(model.state_dict(), f"final_{model_name}_skin_cancer.pth")
    print(f"[SUCCESS] Final model saved as final_{model_name}_skin_cancer.pth")
    print(f"[SUCCESS] Best validation accuracy achieved: {best_val_acc:.4f}")

    # Enhanced plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy plot
    ax1.plot(range(1, num_epochs+1), train_accs, label='Train Accuracy', marker='o', linewidth=2)
    ax1.plot(range(1, num_epochs+1), val_accs, label='Validation Accuracy', marker='s', linewidth=2)
    ax1.axhline(y=best_val_acc, color='r', linestyle='--', alpha=0.8, label=f'Best Val Acc: {best_val_acc:.4f}')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"{model_name.upper()} - Training and Validation Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(range(1, num_epochs+1), train_losses, label='Train Loss', marker='o', linewidth=2)
    ax2.plot(range(1, num_epochs+1), val_losses, label='Validation Loss', marker='s', linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"{model_name.upper()} - Training and Validation Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_training_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n[RESULTS] EfficientNet Training Completed!")
    print(f"[INFO] Model: {model_name.upper()}")
    print(f"[INFO] Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"[INFO] Total Parameters: {total_params:,}")
    print(f"[INFO] Training completed in {num_epochs} epochs")
    print(f"[NOTE] EfficientNet is designed for optimal accuracy/efficiency balance!")
    
    # Compare with previous models
    print(f"\n[COMPARISON] Model Comparison:")
    print(f"  - ResNet152 (Heavy): ~58M params, 88.72% val acc")
    print(f"  - {model_name.upper()}: {total_params:,} params, {best_val_acc:.2%} val acc")
    print(f"  - Efficiency: {model_name} uses {total_params/58158151:.1f}x fewer parameters!")
