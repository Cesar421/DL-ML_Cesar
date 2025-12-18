import pytest
import torch
from torch import nn

from dlcv.utils import load_pretrained_weights

def test_load_pretrained_weights():
    # Set up a simple model
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 5))
    device = torch.device("cpu")
    weights_path = 'saved_models/fixture_weights_dont_delete_this.pth'

    # Ensure model parameters are not already those in the fixture by initializing with random weights
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)

    # Load the pretrained weights into the model
    loaded_model = load_pretrained_weights(model, weights_path, device)

    # Load reference weights to compare
    reference_state_dict = torch.load(weights_path)

    # Check if the weights have been loaded correctly
    for name, param in loaded_model.state_dict().items():
        assert torch.equal(param, reference_state_dict[name]), f"Weights mismatch found for {name}"


