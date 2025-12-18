import pytest
import torch
import numpy as np
import random
from torch import nn
import os

from dlcv.utils import save_model

def test_save_model():
    # Set device and random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = "cpu"

    # Set up a simple model
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 5))
    # Define paths to save the new model and load the reference model
    new_save_path = 'saved_models/test_model'
    reference_model_path = 'saved_models/fixture_weights_dont_delete_this.pth'

    # Save the new model using the function
    save_model(model, new_save_path)

    # Check if the file exists
    assert os.path.exists(new_save_path + ".pth"), "The new model file was not created."
    assert os.path.exists(reference_model_path), "The reference model file does not exist."

    # Load the saved model state_dict and the reference model state_dict
    new_saved_state_dict = torch.load(new_save_path + ".pth")
    reference_state_dict = torch.load(reference_model_path)

    # Compare the new saved model state_dict with the reference model's state_dict
    for param_tensor in model.state_dict():
        assert torch.equal(new_saved_state_dict[param_tensor], reference_state_dict[param_tensor]), "Mismatch in model parameters between the new and reference models."

    # Clean up by removing the file after the test
    os.remove(new_save_path + ".pth")