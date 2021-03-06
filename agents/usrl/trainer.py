import torch
from agents.cnn import CNN


class Trainer:
    def __init__(self, encoder, wandb, device=torch.device("cpu")):
        self.encoder = encoder
        self.wandb = wandb
        self.device = device

    def generate_batch(self, episodes):
        raise NotImplementedError

    def train(self, episodes):
        raise NotImplementedError

    def log_results(self, epoch_idx, epoch_loss, accuracy):
        raise NotImplementedError


class USRLNet(CNN):
    def __init__(self):
        super(USRLNet, self).__init__()
