import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

if __name__ == "__main__":
    # Load dataset with different transforms for train/val
    full_dataset = datasets.ImageFolder(root=data_dir)
    
    # Train/val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(range(len(full_dataset)), [train_size, val_size])
    
    # Create separate datasets with different transforms
    train_dataset = torch.utils.data.Subset(datasets.ImageFolder(root=data_dir, transform=train_transform), train_indices.indices)
    val_dataset = torch.utils.data.Subset(datasets.ImageFolder(root=data_dir, transform=val_transform), val_indices.indices)

    # Data loaders with adjusted batch size for larger models
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Choose a more powerful model - EfficientNet-B3 or ResNet50
    # Option 1: EfficientNet-B3 (more accurate but slower)
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, len(full_dataset.classes))
    )
    
    # Option 2: ResNet50 (good balance of speed and accuracy)
    # model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # model.fc = nn.Sequential(
    #     nn.Dropout(0.3),
    #     nn.Linear(model.fc.in_features, len(full_dataset.classes))
    # )
    
    # Option 3: ResNeXt50 (even more powerful)
    # model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
    # model.fc = nn.Sequential(
    #     nn.Dropout(0.3),
    #     nn.Linear(model.fc.in_features, len(full_dataset.classes))
    # )
    
    model = model.to(device)

    # Enhanced loss and optimizer with learning rate scheduling
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # AdamW with weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # Training loop with more epochs for better convergence
    num_epochs = 25
    train_accs, val_accs, train_losses, val_losses = [], [], [], []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print progress every 50 batches
            if batch_idx % 50 == 0:
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
            torch.save(model.state_dict(), "best_efficientnet_skin_cancer.pth")
            print(f"[SUCCESS] New best model saved with val_acc: {val_acc:.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 80)

    # Save the final model
    torch.save(model.state_dict(), "final_efficientnet_skin_cancer.pth")
    print("[SUCCESS] Final model saved as final_efficientnet_skin_cancer.pth")
    print(f"[SUCCESS] Best validation accuracy achieved: {best_val_acc:.4f}")

    # Enhanced plotting with loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(range(1, num_epochs+1), train_accs, label='Train Accuracy', marker='o')
    ax1.plot(range(1, num_epochs+1), val_accs, label='Validation Accuracy', marker='s')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Training and Validation Accuracy")
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(range(1, num_epochs+1), train_losses, label='Train Loss', marker='o')
    ax2.plot(range(1, num_epochs+1), val_losses, label='Validation Loss', marker='s')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training and Validation Loss")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("enhanced_training_plot.png", dpi=300, bbox_inches='tight')
    plt.show()
