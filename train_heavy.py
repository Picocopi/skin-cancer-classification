import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast

# Device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset path
data_dir = "ham10000_processed"

# Enhanced image transforms for better accuracy
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
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

    # Data loaders - Smaller batch size for heavier model
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Choose heavy model - ResNet152 (60M parameters vs ResNet50's 23M)
    model_choice = "resnet152"  # Options: "resnet152", "wide_resnet101", "resnext101"
    
    if model_choice == "resnet152":
        # ResNet152 - Much heavier and more accurate than ResNet50
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        model_name = "resnet152"
        print("Using ResNet152 - 60M parameters (vs ResNet50's 23M)")
        
    elif model_choice == "wide_resnet101":
        # Wide ResNet101 - Even heavier with wider layers
        model = models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.DEFAULT)
        model_name = "wide_resnet101"
        print("Using Wide ResNet101 - 126M parameters")
        
    elif model_choice == "resnext101":
        # ResNeXt101 - Heavy with grouped convolutions
        model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.DEFAULT)
        model_name = "resnext101"
        print("Using ResNeXt101 - 88M parameters")
    
    # Replace the final layer for our number of classes
    if hasattr(model, 'fc'):
        model.fc = nn.Sequential(
            nn.Dropout(0.4),  # Higher dropout for heavier model
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif hasattr(model, 'classifier'):
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.classifier.in_features, num_classes)
        )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model is {total_params/23522375:.1f}x heavier than ResNet50!")

    # Training setup for heavy model
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)  # Lower LR for stability
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
    
    # Mixed precision for memory efficiency
    scaler = GradScaler()

    # Training parameters
    num_epochs = 12  # Fewer epochs due to heavy model
    train_accs, val_accs, train_losses, val_losses = [], [], [], []
    best_val_acc = 0.0

    print(f"Starting {model_name.upper()} HEAVY training for {num_epochs} epochs...")
    print("Using mixed precision training for memory efficiency...")
    print("-" * 70)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print progress every 25 batches
            if batch_idx % 25 == 0:
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
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{model_name}_heavy.pth")
            print(f"[BEST] New best model saved with val_acc: {val_acc:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
        print("-" * 70)

    # Save the final model
    torch.save(model.state_dict(), f"final_{model_name}_heavy.pth")
    print(f"[SUCCESS] Final model saved as final_{model_name}_heavy.pth")
    print(f"[SUCCESS] Best validation accuracy achieved: {best_val_acc:.4f}")

    # Enhanced plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy plot
    ax1.plot(range(1, num_epochs+1), train_accs, label='Train Accuracy', marker='o', linewidth=2)
    ax1.plot(range(1, num_epochs+1), val_accs, label='Validation Accuracy', marker='s', linewidth=2)
    ax1.axhline(y=best_val_acc, color='r', linestyle='--', alpha=0.8, label=f'Best Val Acc: {best_val_acc:.4f}')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"{model_name.upper()} HEAVY - Training and Validation Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(range(1, num_epochs+1), train_losses, label='Train Loss', marker='o', linewidth=2)
    ax2.plot(range(1, num_epochs+1), val_losses, label='Validation Loss', marker='s', linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"{model_name.upper()} HEAVY - Training and Validation Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_heavy_training_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n[RESULTS] HEAVY Model Training Completed!")
    print(f"[INFO] Model: {model_name.upper()}")
    print(f"[INFO] Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"[INFO] Total Parameters: {total_params:,}")
    print(f"[INFO] Model is {total_params/23522375:.1f}x heavier than ResNet50")
    print(f"[INFO] Training completed in {num_epochs} epochs")
    print(f"[NOTE] This heavy model should achieve higher accuracy than ResNet50!")
