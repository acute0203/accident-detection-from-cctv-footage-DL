import torch
import torch.nn as nn
import torch.optim as optim
from model import DeeperCCTVClassifier
from data import get_dataloaders
from torchvision import models
from rexnet_v1 import ReXNetV1
from torchsummary import summary
import matplotlib.pyplot as plt
import argparse


def get_model(name, num_classes=2):
    if name == 'res':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'rex':
        model = ReXNetV1(classes=num_classes)
    elif name == 'cs':
        model = DeeperCCTVClassifier(num_classes=num_classes)
    else:
        raise ValueError("Unknown model name")
    return model


def main(model_name):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_loader, val_loader = get_dataloaders('../data/train', batch_size=32)
    model = get_model(model_name).to(device)

    print(f"Using model class: {model.__class__.__name__}")
    #summary(model, input_size=(3, 224, 224), device=str(device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_acc = 0.0
    num_epochs = 200
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = f'deep_cctv_model{round(best_val_acc*100,2)}-{model_name}.pth'
            torch.save(model.state_dict(), model_path)
            print("\u2705 Saved new best model!")

    # Plot curves
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title("Loss over Epochs")
    plt.savefig("loss_curve.png")

    plt.figure()
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title("Accuracy over Epochs")
    plt.savefig("acc_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Binary Classification with ResNet, ReXNet or customized model')
    parser.add_argument('--model', type=str, default='cs', choices=['res', 'rex', 'cs'], help='Model name')
    args = parser.parse_args()
    main(args.model)