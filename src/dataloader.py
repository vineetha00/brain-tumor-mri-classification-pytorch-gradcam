from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    train_ds = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_ds = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
    test_ds = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

    return (
        DataLoader(train_ds, batch_size, shuffle=True),
        DataLoader(val_ds, batch_size),
        DataLoader(test_ds, batch_size),
        train_ds.classes
    )
if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_dataloaders("../data", batch_size=8)
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)}")

    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}, Labels: {labels}")
        break
