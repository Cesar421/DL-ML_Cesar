import argparse
import torch
from torch.utils.data import DataLoader

from dlcv.dataset import SubsetSTL10
from dlcv.models import CustomizableNetwork
from dlcv.utils import *
from dlcv.training import train_and_evaluate_model

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # Define transformations for training and testing
    train_transform = get_transforms(
        train=True,
        horizontal_flip_prob=args.horizontal_flip_prob,
        rotation_degrees=args.rotation_degrees
    )
    test_transform = get_transforms(train=False)

    # Load datasets
    train_dataset = SubsetSTL10(
        root=args.data_root,
        split='train',
        transform=train_transform,
        download=True,
        subset_size=args.subset_size
    )
    test_dataset = SubsetSTL10(
        root=args.data_root,
        split='test',
        transform=test_transform,
        download=True
    )

    # Initialize training and test loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Initialize model
    model = CustomizableNetwork(
        conv_layers=args.conv_layers,
        filters_conv1=args.filters_conv1,
        filters_conv2=args.filters_conv2,
        filters_conv3=args.filters_conv3,
        dense_units=args.dense_units
    ).to(device)

    # Load pretrained weights if specified
    if args.pretrained_weights:
        model = load_pretrained_weights(model, args.pretrained_weights, device)
        print(f"Loaded pretrained weights from {args.pretrained_weights}")

    # Optionally freeze layers
    if args.freeze_layers:
        freeze_layers(model, args.freeze_layers)
        print(f"Frozen layers: {args.freeze_layers}")

    # Setup optimizer
    if args.stratification_rates:
        # Use stratified learning rates if specified
        stratification_rates = {}
        # Parse stratification rates from command line
        # Format: layer1:lr1,layer2:lr2
        for item in args.stratification_rates.split(','):
            layer_name, lr = item.split(':')
            stratification_rates[layer_name] = float(lr)
        
        param_groups = get_stratified_param_groups(
            model,
            base_lr=args.base_lr,
            stratification_rates=stratification_rates
        )
        optimizer = torch.optim.Adam(param_groups)
    else:
        # Standard optimizer with single learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)

    # Define the criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Define a scheduler - use the MultiStepLR scheduler
    # Reduces learning rate at specified milestone epochs
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[10, 20, 30, 50, 80],  # Reduce LR at epochs 10, 20, 30, 50, and 80
        gamma=0.1                 # Multiply LR by 0.1 at each milestone
    )
    print(f"Scheduler enabled: Learning rate will be reduced at epochs {[10, 20, 30, 50, 80]}")

    # Hand everything to the train and evaluate model function
    train_losses, test_losses, test_accuracies = train_and_evaluate_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        scheduler=scheduler,
        early_stopping=args.do_early_stopping
    )

    # Save results to CSV
    write_results_to_csv(args.results_csv + "/" + args.run_name, train_losses, test_losses, test_accuracies)
    print(f"Results saved to {args.results_csv}/{args.run_name}.csv")

    # Save the model using the default folder
    if args.save_model_path:
        save_model(model, args.save_model_path + "/" + args.run_name)
        print(f"Model saved to {args.save_model_path}/{args.run_name}.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a customizable CNN on STL-10 dataset.')
    
    # Model architecture arguments
    parser.add_argument('--conv_layers', type=int, default=2, choices=[2, 3],
                        help='Number of convolutional layers (2 or 3)')
    parser.add_argument('--filters_conv1', type=int, default=16,
                        help='Number of filters in the first convolutional layer')
    parser.add_argument('--filters_conv2', type=int, default=32,
                        help='Number of filters in the second convolutional layer')
    parser.add_argument('--filters_conv3', type=int, default=64,
                        help='Number of filters in the third convolutional layer')
    parser.add_argument('--dense_units', type=int, default=128,
                        help='Number of units in the dense layer')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and testing')
    parser.add_argument('--base_lr', type=float, default=0.001,
                        help='Base learning rate for optimizer')
    
    # Data augmentation
    parser.add_argument('--horizontal_flip_prob', type=float, default=0.0,
                        help='Probability of horizontal flip augmentation (0.0 to 1.0)')
    parser.add_argument('--rotation_degrees', type=float, default=0.0,
                        help='Range of rotation augmentation in degrees')
    
    # Transfer learning
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='Path to pretrained weights file')
    parser.add_argument('--freeze_layers', type=str, nargs='+', default=None,
                        help='List of layer names to freeze (e.g., conv1 bn1 conv2)')
    
    # Advanced training options
    parser.add_argument('--stratification_rates', type=str, default=None,
                        help='Stratification rates in format layer1:lr1,layer2:lr2 (e.g., layers.conv1.weight:0.0001,layers.fc1.weight:0.001)')
    parser.add_argument('--do_early_stopping', action='store_true',
                        help='Enable early stopping')
    
    # Dataset options
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for dataset')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Use only a subset of training data (for quick testing)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of worker processes for data loading')
    
    # Output options
    parser.add_argument('--run_name', type=str, required=True,
                        help='Name for this training run (used for saving results and model)')
    parser.add_argument('--results_csv', type=str, default='./results',
                        help='Directory to save training results CSV')
    parser.add_argument('--save_model_path', type=str, default='./saved_models',
                        help='Directory to save trained model')
    
    # Device options
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    args = parser.parse_args()
    main(args)
