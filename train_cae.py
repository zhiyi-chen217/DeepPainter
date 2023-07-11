import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import wandb
import cv2
from Utils import Autoencoder, Encoder, Decoder, ImageDataset, initialize_wandb, EarlyStopping


def main():
    image_dir = "./data/train_cae/image/"
    noise_dir = "./data/train_cae/noise/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        initialize_wandb("cae")

    # Preprocess the image to normalize the pixel values
    preprocess = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
        ]
    )

    # Randomly select 5500 images and discard the rest
    full_dataset = ImageDataset(image_dir, device=device, preprocess=preprocess)
    use_size = 5500
    rest_size = len(full_dataset) - use_size
    use_dataset, _ = torch.utils.data.random_split(full_dataset, [use_size, rest_size])

    # Split into train and validation dataset
    train_size = 5000
    val_size = 500
    train_dataset, val_dataset = torch.utils.data.random_split(use_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32)
    valid_loader = DataLoader(val_dataset, batch_size=32)


    model = Autoencoder(Encoder(), Decoder(), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    n_epoch = 12
    step = 0

    # Check and specify the checkpoint path
    ckpt_path = "/cluster/scratch/zhiychen/DeepPainter/checkpoint" if torch.cuda.is_available() else './checkpoint'
    os.makedirs(ckpt_path, exist_ok=True)
    save_filename = '%s_%s.pth' % (n_epoch, "cae")
    save_path = os.path.join(ckpt_path, save_filename)
    early_stopping = EarlyStopping(save_path=save_path, patience=5, verbose=False, delta=0)

    # Training
    for epoch in range(1, n_epoch + 1):
        train_loss = 0.0
        for data in train_loader:
            input, label = data
            optimizer.zero_grad()
            # temp_1 = input[0, :, :, :].permute(1, 2, 0).detach().numpy()
            # cv2.imshow("image", temp_1)
            # cv2.waitKey(0)

            output = model(input)

            # temp_2 = output[0, :, :, :].permute(1, 2, 0).detach().numpy()
            # cv2.imshow("image", temp_2)
            # cv2.waitKey(0)

            loss = criterion(output, label)

            loss.backward()
            optimizer.step()
            step += 1
            train_loss += loss.item()
            # logging
            wandb.log({"Train_loss": loss.item()})
        print({"Total_Train_loss": train_loss})

        # Validation after each epoch
        total_loss, batch_count = 0, 0
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                input, label = batch
                output = model(input)
                loss = criterion(output, label)
                batch_count += len(input)
                total_loss += loss.item() * len(input)
        valid_loss = total_loss / batch_count
        wandb.log({"Valid_loss": valid_loss, "Epochs": epoch})

        # for early stopping
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
main()
