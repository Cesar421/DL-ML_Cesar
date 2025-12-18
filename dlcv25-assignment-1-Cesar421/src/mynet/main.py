import torch
import torch.nn as nn
import torch.optim as optim

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
        float: The average loss for the entire epoch.
    """
    # Set the model to training mode
    model.train()
    
    # Initialize running loss
    running_loss = 0.0
    #-----i=0
    # Iterate over the data loader
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #-----print(f"running_loss: {running_loss} and {i}")
        #-----i+=1
    # Calculate average loss for the epoch
    average_loss = running_loss / len(data_loader)
    
    return average_loss;


def evaluate_one_epoch(model, data_loader, criterion, device):
    """
    Tests a given model for one epoch using the provided data loader and criterion.

    Args:
        model (nn.Module): The model to be tested.
        data_loader (DataLoader): The data loader providing the testing data.
        criterion (nn.Module): The loss function to be used during testing.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').

    Returns:
        float: The average loss for the entire epoch.
        float: The accuracy of the model on the test data.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize running loss and correct predictions counter
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Iterate over the data loader
        for images, labels in data_loader:
            # Move data to the specified device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Accumulate loss
            running_loss += loss.item()
            
            # Get predictions (class with highest probability)
            _, predicted = torch.max(outputs.data, 1)
            
            # Update total samples and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate average loss for the epoch
    average_loss = running_loss / len(data_loader)
    
    # Calculate accuracy
    accuracy = correct / total
    
    return average_loss, accuracy;

def train_and_evaluate_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, scheduler=None):
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
        list: A list of the average loss for each epoch.
        list: A list of the average loss for each testing epoch.
        list: A list of the accuracy for each testing epoch.
    """
    # Initialize lists to store metrics
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    # Training loop for each epoch
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Evaluate on test set
        test_loss, test_accuracy = evaluate_one_epoch(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # Step the scheduler if provided
        if scheduler is not None:
            scheduler.step()
        
        # print(f'Epoch [{epoch+1}/{num_epochs}], '
        #       f'Train Loss: {train_loss:.4f}, '
        #       f'Test Loss: {test_loss:.4f}, '
        #       f'Test Accuracy: {test_accuracy:.4f}')
    
    return train_losses, test_losses, test_accuracies

if __name__ == "__main__":

    import sys
    import os
    from torch.utils.data import DataLoader
    from torchvision import transforms
    
    # Add the parent directory to sys.path to access mynet module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Import after adding to path
    from mynet.dataset import CustomMNISTDataset
    from mynet.model import ThreeLayerFullyConnectedNetwork
    from mynet.visualize import visualize_results, plot_training_history
    
    # Configure transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])


    # Get the correct path to the data (go up 2 levels from src/mynet/)
    project_root = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    data_path = os.path.join(project_root, 'data')
    images_path = os.path.join(project_root, 'images')

    print("="*60)
    print("MNIST Training and Evaluation Pipeline")
    print("="*60)
    print(f"Project root: {project_root}")
    print(f"Data root: {data_path}")
    print(f"Images output: {images_path}")
    print("="*60)

    # Load training dataset
    train_dataset = CustomMNISTDataset(root=data_path, subset='train', transformation=transform)
    print(f"\nTraining dataset loaded: {len(train_dataset)} samples")
    
    # Load test dataset
    test_dataset = CustomMNISTDataset(root=data_path, subset='test', transformation=transform)
    print(f"Test dataset loaded: {len(test_dataset)} samples")

    # Create DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"DataLoaders created (batch_size={batch_size})")

    # Initialize model
    model = ThreeLayerFullyConnectedNetwork()
    print(f"\nModel initialized: ThreeLayerFullyConnectedNetwork")
    print("\nModel Architecture:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")

    # Define optimizer and loss function
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    print(f"\nOptimizer: SGD (lr={learning_rate}, momentum=0.9)")
    print(f"Loss function: CrossEntropyLoss")
    print(f"Scheduler: StepLR (step_size=30, gamma=0.1)")

    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Device: {device}")

    # Training parameters
    num_epochs = 10
    print(f"\n{'='*60}")
    print(f"Starting training for {num_epochs} epochs...")
    print(f"{'='*60}\n")

    # Train and evaluate the model
    train_losses, test_losses, test_accuracies = train_and_evaluate_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        scheduler=scheduler
    )

    # Print training results
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print("\nEpoch-by-Epoch Results:")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Test Loss':<12} {'Test Accuracy':<15}")
    print("-" * 60)
    for epoch in range(num_epochs):
        print(f"{epoch+1:<8} {train_losses[epoch]:<12.4f} {test_losses[epoch]:<12.4f} {test_accuracies[epoch]:<15.4f}")
    
    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"  Best Test Accuracy: {max(test_accuracies):.4f} (Epoch {test_accuracies.index(max(test_accuracies))+1})")
    print(f"  Final Train Loss: {train_losses[-1]:.4f}")
    print(f"  Final Test Loss: {test_losses[-1]:.4f}")
    print(f"  Final Test Accuracy: {test_accuracies[-1]:.4f}")
    print(f"{'='*60}\n")

    # Create and save training history plot
    print("Creating visualizations...")
    plot_path = os.path.join(images_path, 'training_history.png')
    plot_training_history(train_losses, test_losses, test_accuracies, save_path=plot_path)
    print(f"Training history plot saved to: {plot_path}")

    # Create and save inference visualization
    print("\nGenerating inference predictions on test data...")
    class_names = [str(i) for i in range(10)]  # MNIST digits 0-9
    fig = visualize_results(model, test_loader, num_images=10, class_names=class_names, device=device)
    inference_path = os.path.join(images_path, 'inference_predictions.png')
    fig.savefig(inference_path, dpi=300, bbox_inches='tight')
    print(f"Inference predictions saved to: {inference_path}")

    # Save the trained model (optional)
    model_path = os.path.join(project_root, 'trained_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to: {model_path}")

    print(f"\n{'='*60}")
    print("All tasks completed successfully!")
    print(f"{'='*60}")
