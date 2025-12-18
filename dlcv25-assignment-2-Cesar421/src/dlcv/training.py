import torch
from tqdm import tqdm

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    """
    Trains a given model for one epoch using the provided data loader, criterion, and optimizer.

    Args:
        model (nn.Module): The model to be trained.
        data_loader (DataLoader): The data loader providing the training data.
        criterion (nn.Module): The loss function to be used during training.
        optimizer (torch.optim.Optimizer): The optimizer to be used for updating the model's parameters.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').

    Returns:
        float: The average loss per batch for the entire epoch.
    """
    model.train()
    running_loss = 0.0
    total_batches = len(data_loader)

    for inputs, labels in tqdm(data_loader, desc="Training", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / total_batches
    return average_loss

def evaluate_one_epoch(model, data_loader, criterion, device):
    """
    Tests a given model for one epoch using the provided data loader and criterion.

    Args:
        model (nn.Module): The model to be tested.
        data_loader (DataLoader): The data loader providing the testing data.
        criterion (nn.Module): The loss function to be used during testing.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').

    Returns:
        float: The average loss per batch for the entire epoch.
        float: The accuracy of the model on the test data.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    total_batches = len(data_loader)
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    average_loss = running_loss / total_batches
    accuracy = correct_predictions / total_samples
    return average_loss, accuracy

def train_and_evaluate_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, scheduler=None, early_stopping=False):
    """
    Trains a given model for a specified number of epochs using the provided data loader, criterion,
    and optimizer, and tracks the loss for each epoch.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The data loader providing the training data.
        test_loader (DataLoader): The data loader providing the testing data.
        criterion (nn.Module): The loss function to be used during training and testing.
        optimizer (torch.optim.Optimizer): The optimizer to be used for updating the model's parameters.
        num_epochs (int): The number of epochs to train the model.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').
        scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler (default is None).

    Returns:
        list: A list of the average loss per batch for each epoch.
        list: A list of the average loss per batch for each testing epoch.
        list: A list of the accuracy for each testing epoch.
    """
    train_losses = []
    test_losses = []
    test_accuracies = []
    best_test_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate_one_epoch(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        if scheduler:
            scheduler.step()

        # Early stopping: stop if test loss doesn't improve for 2 consecutive epochs
        if early_stopping:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                epochs_no_improve = 0
                print(f"  → Test loss improved to {best_test_loss:.4f}")
            else:
                epochs_no_improve += 1
                print(f"  → No improvement for {epochs_no_improve} epoch(s)")
                if epochs_no_improve >= 2:
                    print(f"Early stopping triggered after {epoch + 1} epochs (no improvement for 2 consecutive epochs)")
                    break
    return train_losses, test_losses, test_accuracies