import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from dlcv.utils import plot_multiple_losses_and_accuracies

def load_csv_data(csv_path):
    """Load training data from CSV file."""
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_losses.append(float(row['Train Loss']))
            test_losses.append(float(row['Test Loss']))
            test_accuracies.append(float(row['Test Accuracy']))
    
    return train_losses, test_losses, test_accuracies

def main(args):
    """Load multiple training runs and plot comparisons."""
    model_data_list = []
    
    # Load each CSV file
    for csv_file in args.csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"Warning: {csv_file} not found, skipping...")
            continue
        
        # Extract model name from filename
        model_name = csv_path.stem
        
        # Load data
        train_losses, test_losses, test_accuracies = load_csv_data(csv_path)
        
        model_data_list.append({
            'name': model_name,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies
        })
    
    if not model_data_list:
        print("No valid CSV files found!")
        return
    
    # Create comparison plot
    output_path = Path(args.output_dir) / args.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_multiple_losses_and_accuracies(model_data_list)
    
    print(f"Comparison plot saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare multiple training runs.')
    parser.add_argument('csv_files', type=str, nargs='+',
                        help='Paths to CSV files with training results')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save the plot')
    parser.add_argument('--output_name', type=str, default='comparison_plot.png',
                        help='Name of the output plot file')
    
    args = parser.parse_args()
    main(args)