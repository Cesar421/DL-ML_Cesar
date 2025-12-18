import csv
import os
from pathlib import Path

import torch
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# STL10 normalization statistics (pre-computed)
STL10_MEAN = [0.4467, 0.4398, 0.4066]
STL10_STD = [0.2603, 0.2566, 0.2713]

def load_pretrained_weights(network, weights_path, device):
    """
    Loads pretrained weights (state_dict) into the specified network.

    Args:
        network (nn.Module): The network into which the weights are to be loaded.
        weights_path (str or pathlib.Path): The path to the file containing the pretrained weights.
        device (torch.device): The device on which the network is running (e.g., 'cpu' or 'cuda').
    Returns:
        network (nn.Module): The network with the pretrained weights loaded and adjusted if necessary.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found at {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)

    # Some checkpoints might wrap the actual state_dict
    state_dict = checkpoint.get('state_dict', checkpoint)

    model_state = network.state_dict()
    filtered_state = {}
    for name, param in state_dict.items():
        if name in model_state and model_state[name].shape == param.shape:
            filtered_state[name] = param

    model_state.update(filtered_state)
    network.load_state_dict(model_state)
    network.to(device)
    return network

def freeze_layers(network, frozen_layers):
    """
    Freezes the specified layers of a network. Freezing a layer means its parameters will not be updated during training.

    Args:
        network (nn.Module): The neural network to modify.
        frozen_layers (list of str): A list of layer identifiers whose parameters should be frozen.
    """
    if not frozen_layers:
        return

    normalized = set(frozen_layers)
    for name, param in network.named_parameters():
        if any(name.startswith(layer_name) for layer_name in normalized):
            param.requires_grad = False
        else:
            param.requires_grad = True

def save_model(model, path):
    """
    Saves the model state_dict to a specified file.

    Args:
        model (nn.Module): The PyTorch model to save. Only the state_dict should be saved.
        path (str): The path where to save the model. Without the postifix .pth
    """
    path = Path(path)
    if path.suffix != '.pth':
        path = path.with_suffix('.pth')

    if path.parent:
        os.makedirs(path.parent, exist_ok=True)

    torch.save(model.state_dict(), path)

def get_stratified_param_groups(network, base_lr=0.001, stratification_rates=None):
    """
    Creates parameter groups with different learning rates for different layers of the network.

    Args:
        network (nn.Module): The neural network for which the parameter groups are created.
        base_lr (float): The base learning rate for layers not specified in stratification_rates.
        stratification_rates (dict): A dictionary mapping layer names to specific learning rates.

    Returns:
        param_groups (list of dict): A list of parameter group dictionaries suitable for an optimizer.
                                     Outside of the function this param_groups variable can be used like:
                                     optimizer = torch.optim.Adam(param_groups)
    """
    stratification_rates = stratification_rates or {}
    param_groups = []
    assigned_param_ids = set()

    for layer_name, lr in stratification_rates.items():
        params = [param for name, param in network.named_parameters() if name.startswith(layer_name)]
        if params:
            param_groups.append({'params': params, 'lr': lr})
            assigned_param_ids.update(id(p) for p in params)

    remaining_params = [param for param in network.parameters() if id(param) not in assigned_param_ids]
    if remaining_params:
        param_groups.append({'params': remaining_params, 'lr': base_lr})

    return param_groups

def get_transforms(train=True, horizontal_flip_prob=0.0, rotation_degrees=0.0):
    """
    Creates a torchvision transform pipeline for training and testing datasets. For training, augmentations
    such as horizontal flipping and random rotation can be included. For testing, only essential transformations
    like normalization and converting the image to a tensor are applied.

    Args:
        train (bool): Indicates whether the transform is for training or testing. If True, augmentations are applied.
        horizontal_flip_prob (float): Probability of applying a horizontal flip to the images. Effective only if train=True.
        rotation_degrees (float): The range of degrees for random rotation. Effective only if train=True.

    Returns:
        torchvision.transforms.Compose: Composed torchvision transforms for data preprocessing.
    """
    transform_list = []

    if train:
        if horizontal_flip_prob > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=horizontal_flip_prob))
        if rotation_degrees > 0:
            transform_list.append(transforms.RandomRotation(rotation_degrees))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=STL10_MEAN, std=STL10_STD)
    ])

    return transforms.Compose(transform_list)

def write_results_to_csv(file_path, train_losses, test_losses, test_accuracies):
    """
    Writes the training and testing results to a CSV file.

    Args:
        file_path (str): Path to the CSV file where results will be saved. Without the postfix .csv
        train_losses (list): List of training losses.
        test_losses (list): List of testing losses.
        test_accuracies (list): List of testing accuracies.
    """
    with open(file_path + ".csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Test Accuracy'])
        for epoch in range(len(train_losses)):
            writer.writerow([epoch + 1, train_losses[epoch], test_losses[epoch], test_accuracies[epoch]])

def plot_multiple_losses_and_accuracies(model_data_list):
    """
    Plots training loss, test loss, and test accuracy for multiple models.
    Train and test loss share the same plot.
    
    Args:
        model_data_list (list of dict): List of dictionaries, each containing:
            - 'name': Model name (str)
            - 'train_losses': List of training losses
            - 'test_losses': List of test losses
            - 'test_accuracies': List of test accuracies
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for model_data in model_data_list:
        epochs = range(1, len(model_data['train_losses']) + 1)
        
        # Plot training loss and test loss on the same plot
        axes[0].plot(epochs, model_data['train_losses'],
                     marker='o', linestyle='-', 
                     label=f"{model_data['name']} (Train)")
        axes[0].plot(epochs, model_data['test_losses'], 
                     marker='s', linestyle='--', 
                     label=f"{model_data['name']} (Test)")
        
        # Plot test accuracy
        axes[1].plot(epochs, model_data['test_accuracies'], 
                     marker='o', label=model_data['name'])
    
    # Configure plot 1: Training and Test Loss
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Configure plot 2: Test Accuracy
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Test Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('./results/comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Comparison plot saved to ./results/comparison_plot.png")

def plot_samples_with_predictions(images, labels, predictions, class_names):
    """
    Plots a grid of images with labels and predictions, with dynamically adjusted text placement.

    Args:
        images (Tensor): Batch of images.
        labels (Tensor): True labels corresponding to the images.
        predictions (Tensor): Predicted labels for the images.
        class_names (list): List of class names indexed according to labels.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Denormalize images for display
    mean = torch.tensor(STL10_MEAN).view(3, 1, 1)
    std = torch.tensor(STL10_STD).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    # Show up to 16 images
    n_images = min(16, len(images))
    grid_size = int(np.ceil(np.sqrt(n_images)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for idx in range(n_images):
        # Convert from CHW to HWC for matplotlib
        img = images[idx].permute(1, 2, 0).numpy()
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Get labels
        true_label = class_names[labels[idx]]
        pred_label = class_names[predictions[idx]]
        
        # Color: green if correct, red if wrong
        color = 'green' if labels[idx] == predictions[idx] else 'red'
        
        # Add text
        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}', 
                           fontsize=10, color=color, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/sample_predictions_Overfitting.png', dpi=150, bbox_inches='tight')
    print("Sample predictions saved to: results/sample_predictions.png")
    plt.show()


def plot_confusion_matrix(labels, preds, class_names):
    """
    Plots a confusion matrix using ground truth labels and predictions.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import numpy as np
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # Normalize to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Display confusion matrix
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, label='Percentage (%)')
    
    # Set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix (%)')
    
    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Show count and percentage
            text = f'{cm[i, j]}\n({cm_normalized[i, j]:.1f}%)'
            ax.text(j, i, text,
                   ha="center", va="center",
                   color="white" if cm_normalized[i, j] > thresh else "black",
                   fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrix_overfitting.png', dpi=150, bbox_inches='tight')
    print("Confusion matrix saved to: results/confusion_matrix.png")
    plt.show()
