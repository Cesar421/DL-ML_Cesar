from torchvision.datasets import MNIST
import os
from pathlib import Path
import csv
import sys
import argparse

def save_dataset(dataset, data_location, set_name):
    foldername = data_location / "MNIST" / set_name
    foldername.mkdir(parents=True, exist_ok=True)
    csv_path = data_location / "MNIST" / f"{set_name}.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'label', 'named_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (data, label) in enumerate(dataset):
            named_label = dataset.classes[label]
            image_path = foldername / f"{idx}.png"
            data.save(image_path)
            writer.writerow({'filename': f"{idx}.png", 'label': label, 'named_label': named_label})

def load_and_create_datasets(data_location, logging_off):
    print("Downloading and creating MNIST dataset...")
    if logging_off:
        f = open(os.devnull, "w")
        sys.stdout = f
    data_location = Path(data_location)
    print(f"Storing data at {data_location.resolve()}.")

    train_data = MNIST(root=data_location, train=True, download=True)
    test_data = MNIST(root=data_location, train=False, download=True)

    save_dataset(train_data, data_location, "train")
    save_dataset(test_data, data_location, "test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download MNIST from PyTorch and move the raw data into a folder.")
    parser.add_argument("dir", type=str, help="Folder to store images.")
    parser.add_argument("--logging_off", action="store_true", help="Turn off logging.")
    parser.set_defaults(logging_off=False)
    args = parser.parse_args()
    load_and_create_datasets(args.dir, args.logging_off)
