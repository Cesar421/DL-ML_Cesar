import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_results(model, dataloader, num_images, class_names, device):
    """
    Visualizes the results of a neural network by plotting the images with their ground truth
    and predictions side by side.

    Args:
        model (nn.Module): The trained neural network model.
        dataloader (DataLoader): The DataLoader providing the images and labels.
        num_images (int): The number of images to display.
        class_names (list): A list of class names corresponding to the dataset class indices.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').
    """
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(dataloader))
    images = images[:num_images].to(device)
    labels = labels[:num_images]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    # Move data to CPU for plotting
    images = images.cpu()
    predictions = predictions.cpu()
    
    # Create subplot grid
    fig, axes = plt.subplots(2, num_images // 2, figsize=(15, 6))
    axes = axes.flatten()
    
    for idx in range(num_images):
        # Get image and reshape if needed
        img = images[idx].squeeze()
        
        # Plot image
        axes[idx].imshow(img, cmap='gray')
        
        # Set title with ground truth and prediction
        true_label = class_names[labels[idx]]
        pred_label = class_names[predictions[idx]]
        color = 'green' if labels[idx] == predictions[idx] else 'red'
        
        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}', 
                           color=color, fontsize=10)
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_training_history(train_losses, test_losses, test_accuracies, save_path=None):
    """
    Plots the training and test losses, along with test accuracy over epochs.
    
    Args:
        train_losses (list): List of training losses per epoch.
        test_losses (list): List of test losses per epoch.
        test_accuracies (list): List of test accuracies per epoch.
        save_path (str, optional): Path to save the figure. If None, displays the plot.
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Test Loss over Epochs', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, test_accuracies, 'g-', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Test Accuracy over Epochs', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    return fig
