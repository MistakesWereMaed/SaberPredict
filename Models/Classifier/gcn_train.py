import torch

import numpy as np

from tqdm import tqdm
from torch_geometric.nn.models import GCN
from torch_geometric.loader import DataLoader
from dataloaders.gcn_dataloader import sabre_dataset
from torch.utils.data import random_split


def load_model():
    pass

def load_train_dataloader(**kwargs):
    data_fl = kwargs.get("data_fl", "/Users/johannesbauer/Documents/Coding/SaberPredict/data/test_2.yaml")
    train_split = kwargs.get("train_split", 0.8)
    
    batches = kwargs.get("batches", 30)

    ds = sabre_dataset(data_fl)

    if train_split <= 1:

        train_ds, val_ds = random_split(ds, [0.8, 0.2])

        train_dl = DataLoader(train_ds, batch_size=batches, shuffle = True)
        val_dl = DataLoader(val_ds, batch_size = batches, shuffle = False)

    else:
        train_dl = DataLoader(ds, batch_size=batches, shuffle = False)
        val_dl = None

    return train_dl, val_dl

def get_device():

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("MPS")

    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print()

    return device

def validation_func(model, validation_dataloader, loss, device, epoch):
    model.eval()

    n_batches = 0
    total_loss = 0

    for batch in validation_dataloader:
        batch_x = batch.x
        edge_vals = batch.edge_index
        
        batch_x = batch_x.to(device)
        edge_vals = edge_vals.to(device)

        y = batch.y

        out = model(x = batch_x, edge_index = edge_vals)

        loss_for_iter = loss(out, y)

        total_loss += loss_for_iter.item()
        n_batches += 1

    average_loss = total_loss/n_batches

    print(f"Epoch: {epoch}, Loss: {average_loss:.5f}")

    return epoch


def train_model(md, train_dataloader, optimizer, loss, val_dataloader=None, **kwargs):
    
    in_channels = kwargs.get("in_channels", -1)
    out_channels = kwargs.get("out_channels", 10)
    hidden_layers = kwargs.get("hidden_layers", 3)
    num_layers = kwargs.get("num_layer", 2)
    epochs = kwargs.get("epochs", 50)

    graph_gcn = GCN(in_channels=in_channels,
                    out_channels=out_channels,
                    hidden_layers=hidden_layers,
                    num_layers=num_layers
                    )


    device = get_device()

    graph_gcn.to(device)

    for epoch in tqdm(range(epochs), desc = "Epoch"):
        graph_gcn.train()

        for n, batch in enumerate(train_dataloader):


            batch_x = batch.x
            edge_vals = batch.edge_index
            
            batch_x = batch_x.to(device)
            edge_vals = edge_vals.to(device)

            y = batch.y

            optimizer.zero_grad()
            out = graph_gcn(x = batch_x, edge_index = edge_vals)

            loss_for_iter = loss(out, y)

            loss.backward()
            optimizer.step()

            if n % 50:
                print(f"Epoch: {epoch}, loss: {loss_for_iter}")

        if val_dataloader is not None:
            _ = validation_func(graph_gcn, val_dataloader, loss, device, epoch)

    return graph_gcn

def test_model():
    pass

def main():
    pass


if __name__ == "__main__":
    main