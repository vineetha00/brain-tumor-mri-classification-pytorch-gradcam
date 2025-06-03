import torch
from torch import nn, optim
from src.dataloader import get_dataloaders
from src.model import get_model
from src.train import train

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, val_loader, test_loader, class_names = get_dataloaders("data", batch_size=32)

    model = get_model(num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train(model, train_loader, val_loader, criterion, optimizer, device)

    # Save the model after training
    import os
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/resnet18_brain_mri.pt")

if __name__ == "__main__":
    main()



