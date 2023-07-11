import json
import os

import cv2
import torch
import wandb
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, in_channels=3, act_fn=nn.ReLU()):
        super().__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, 100, 5, padding=2),
            act_fn, )
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(100, 200, 5, padding=2),
            act_fn,
        )

    def forward(self, x):
        output = self.conv_1(x)  # (3, 256, 256) -> (100, 256, 256)
        output, indices_1 = self.pool(output)  # (100, 256, 256) -> (100, 128, 128)
        output = self.conv_2(output)  # (100, 128, 128) -> (200, 128, 128)
        output, indices_2 = self.pool(output)  # (200, 128, 128) -> (200, 64, 64)
        return output, indices_1, indices_2


#  defining decoder
class Decoder(nn.Module):
    def __init__(self, act_fn=nn.ReLU()):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(200, 100, 5, padding=2),
            act_fn, )
        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(100, 3, 5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x, indices_1, indices_2):
        output = self.unpool(x, indices_1)  # (200, 64, 64) -> (200, 128, 128)
        output = self.deconv_1(output)  # (200, 128, 128) -> (100, 128, 128)
        output = self.unpool(output, indices_2) # (100, 128, 128) -> (200, 256, 256)
        output = self.deconv_2(output) # (100, 256, 256) -> (3, 256, 256)
        return output


#  defining autoencoder
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.encoder.to(device)

        self.decoder = decoder
        self.decoder.to(device)

    def forward(self, x):
        encoded, indices_2, indices_1 = self.encoder(x) # returning the pooling indices as well
        decoded = self.decoder(encoded, indices_1, indices_2)
        return decoded


class CNNClf(nn.Module):
    def __init__(self, encoder,  device, act_fn=nn.ReLU()):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():  # freeze the feature extractor
            param.requires_grad = False

        self.encoder.to(device)

        self.fc = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(200*64*64, 400),
            act_fn,
            nn.Linear(400, 200),
            act_fn,
            nn.Linear(200, 3),
        )

    def forward(self, x):
        encoded, _, _ = self.encoder(x)
        output = self.fc(encoded)
        return output


class ImageDataset(Dataset):

    def __init__(self, image_dir, device, preprocess, test=False):
        self.image_dir = image_dir
        self.test = test
        self.device = device
        self.preprocess = preprocess

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, "image_" + str(idx) + ".jpg")
        image = read_image(image_path)
        image = torch.tensor(image)
        image = image[:3, :, :]

        # Works with 3 channels
        mask = np.random.choice([0, 1], size=(256, 256), p=[.2, .8]).astype(np.uint8)
        noise = cv2.bitwise_and(image.permute(1, 2, 0).numpy(), image.permute(1, 2, 0).numpy(), mask=mask)
        noise = torch.tensor(noise)
        noise = noise.permute(2, 0, 1)
        noise = noise[:3, :, :]

        image = self.preprocess(image)
        noise = self.preprocess(noise)


        return noise.to(self.device), image.to(
            self.device)

class ClfImageDataset(Dataset):

    def __init__(self, image_dir, labels, device, preprocess, test=False):
        self.image_dir = image_dir
        self.labels = labels
        self.test = test
        self.device = device
        self.preprocess = preprocess

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, "image_" + str(idx) + ".jpg")
        image = read_image(image_path)
        image = image[:3, :, :]  # Works with 3 channels
        image = self.preprocess(image)

        image = torch.tensor(image)
        label = torch.tensor(self.labels[idx, :])
        return image.to(self.device), label.to(self.device)


# for logging purposes
def initialize_wandb(wandb_name):
    log_directory = os.getcwd()

    print(f'log_directory: {log_directory}')
    wandb_name = f'{wandb_name}'
    wandb.init(project='DeepPainter', name=wandb_name, notes='', dir=log_directory,
               settings=wandb.Settings(start_method='fork'), mode="online")
    args_to_log = dict()
    args_to_log['out_dir'] = log_directory
    print("\n" + json.dumps(str(args_to_log), indent=4) + "\n")
    wandb.config.update(args_to_log)
    del args_to_log

# for computing accuracy
def compute_n_correct(labels, output):
    cat_labels = np.argmax(labels, axis=1)
    cat_outputs = np.argmax(output, axis=1)
    return np.sum((cat_outputs == cat_labels).numpy())


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=5, verbose=False, delta=0, save=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        self.save = save

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.save:
                self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save:
                self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)	# save the current best model
        self.val_loss_min = val_loss
