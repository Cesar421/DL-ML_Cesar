from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image

class CustomMNISTDataset( Dataset ):
    """
    A custom dataset class for MNIST-like datasets that interfaces with PyTorch's Dataset. This class
    facilitates loading and transforming images from a specified subset (e.g., 'train' or 'test'),
    where the paths and labels for the images are provided in a CSV file.

    Methods (besides the constructor):
    __getitem__(self, idx):
        Retrieves an image and its label at the specified index `idx`, optionally applies
        transformations, and returns the transformed image and its label.

    __len__(self):
        Returns the total number of images in the dataset.
    """

    def __init__(self, root: str, subset: str, transformation=None):
        """
        Initializes the dataset object, setting up the directory paths, loading image paths and labels
        from a CSV file.

        Args:
            root (str): Path to the root directory where the images and CSV file are stored.
            subset (str): Identifier for the subset being used (e.g., 'train', 'test'). This is used for finding the
                          image folder and the .csv file containing the annoations.
            transformation (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.subset = subset
        self.transformation = transformation
        #print(f"Iniciando CustomMNISTDataset con root: {root}, subset: {subset}")
        
        # Handle different root path formats
        if root.endswith("MNIST"):
            # If root already includes MNIST path (like "data/MNIST")
            self.csv_file = os.path.join(root, f"{subset}.csv")
            self.img_dir = os.path.join(root, subset)
        else:
            # If root is just the base path (like "data")
            self.csv_file = os.path.join(root, "MNIST", f"{subset}.csv")
            self.img_dir = os.path.join(root, "MNIST", subset)
        
        # Load the CSV file into a pandas DataFrame
        self.annotations = pd.read_csv(self.csv_file)
        
        # Extract filenames and labels
        self.image_names = self.annotations['filename'].tolist()
        self.labels = self.annotations['label'].tolist()

    def __getitem__(self, idx):
        """
        Retrieves an image and its label at the specified index `idx`, optionally applies
        transformations, and returns the transformed image and its label.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            tuple: (image, label) where image is the transformed image and label is the class label.
        """
        #print(f"Fetching item at index: {idx}")
        # Get the image filename and label for the given index
        img_name = self.image_names[idx]
        label = self.labels[idx]
        
        # Construct the full path to the image
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load the image using PIL
        image = Image.open(img_path)
        
        # Apply transformations if provided
        if self.transformation:
            image = self.transformation(image)
            
        return image, label

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        
        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.annotations)