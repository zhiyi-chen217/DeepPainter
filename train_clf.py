import os

import numpy as np
import torch
import wandb
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision.transforms import transforms

from Utils import Autoencoder, Encoder, Decoder, ClfImageDataset, initialize_wandb, CNNClf, compute_n_correct, \
    EarlyStopping


def main():
    image_dir = "./data/train_clf"
    label_dir = "./data/labels.csv"
    # Specify which checkpoint to use
    cae_ckpt_path = "checkpoint/10_cae.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        initialize_wandb("clf")
    preprocess = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
        ]
    )


    labels = np.loadtxt(label_dir, delimiter=",")
    full_dataset = ClfImageDataset(image_dir, labels, device=device, preprocess=preprocess)

    # Use the pretrained Convolutional AutoEncoder
    cae_model = Autoencoder(Encoder(), Decoder(), device)
    cae_model.load_state_dict(torch.load(cae_ckpt_path, map_location=device))

    # Construct the Classifier model
    model = CNNClf(cae_model.encoder, device=device)
    print(model)
    model.to(device)

    # Split into training and testing dataset
    train_size = int(0.9*len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_size = int(0.9*len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32)
    valid_loader = DataLoader(val_dataset, batch_size=32)

    test_loader = DataLoader(test_dataset, batch_size=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    n_epoch = 20
    step = 0
    ckpt_path = "/cluster/scratch/zhiychen/DeepPainter/checkpoint" if torch.cuda.is_available() else './checkpoint'
    os.makedirs(ckpt_path, exist_ok=True)
    save_filename = '%s_%s.pth' % (n_epoch, "clf")
    save_path = os.path.join(ckpt_path, save_filename)
    early_stopping = EarlyStopping(save_path=save_path, patience=5, verbose=False, delta=0)

    for epoch in range(1, n_epoch + 1):
        train_loss = 0.0
        for data in train_loader:
            input, label = data
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()
            step += 1
            train_loss += loss
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

    # Testing here
    model = CNNClf(cae_model.encoder, device=device)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)
    
    model.eval()
    with torch.no_grad():
        n_correct = 0
        for data in test_loader:
            input, label = data
            output = model(input)
            n_correct += compute_n_correct(label.cpu(), output.cpu())
    print({"Total_accuracy": n_correct / len(test_loader.dataset)})

main()
