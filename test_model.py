import json
import os

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
import torch.nn as nn
import wandb

from Utils import Autoencoder, Decoder, Encoder, ImageDataset


def main():
    image_dir = "./data/train_cae/image/"
    noise_dir = "./data/train_cae/noise/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        initialize_wandb()
    preprocess = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
        ]
    )
    model = Autoencoder(Encoder(), Decoder(), device)
    full_dataset = ImageDataset(image_dir, noise_dir, device=device, preprocess=preprocess)

    train_size = 5000
    val_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    n_epoch = 5
    step = 0
    ckpt_path = "checkpoint/5_cae.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))

    for epoch in range(1, n_epoch + 1):
        train_loss = 0.0
        for data in train_loader:
            input, label = data
            optimizer.zero_grad()
            temp_1 = input[0, :, :, :].permute(1, 2, 0).detach().numpy()
            cv2.imshow("image", temp_1)
            cv2.waitKey(0)

            output = model(input)

            temp_2 = output[0, :, :, :].permute(1, 2, 0).detach().numpy()
            cv2.imshow("image", temp_2)
            cv2.waitKey(0)

            loss = criterion(output, label)

            loss.backward()
            optimizer.step()
            step += 1
            train_loss += loss
            # logging
            print({"Train_loss": loss})
        print({"Total_Train_loss": train_loss})
main()