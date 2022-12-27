import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm as prog
from numpy import savetxt
import statistics as s
import numpy as np

from builder.dataset_provider_combined import (
    get_loader
)

from pytorchtools import EarlyStopping

from torchmetrics.regression import MeanSquaredError
from torchmetrics.image import StructuralSimilarityIndexMeasure

import shutup
shutup.please()

from model.model_combined import UNET
from configuration import (
    device,
    path_train,
    path_validation,
    batch_size,
    num_workers,
    pin_memory
)


def train(epoch, loader, loss_fn, optimizer, scaler, model, mse, ssim):
    torch.enable_grad()
    model.train()

    loop = prog(loader)

    running_loss = []
    running_mae = []
    running_mse = []
    running_ssim = []

    for batch_index, (data, target) in enumerate(loop):
        optimizer.zero_grad(set_to_none=True)
        data = data.to(device)

        data[data < 0] = 0
        target[target < 0] = 0

        data = model(data)
        data[data < 0] = 0

        target = target.to(device).unsqueeze(1)

        with torch.cuda.amp.autocast():
            loss = loss_fn(data, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        loss_value = loss.item()

        running_mae.append(loss_value)
        running_mse.append(mse(data, target).item())
        running_ssim.append(ssim(data, target).item())

        mse.reset()
        ssim.reset()

        loop.set_postfix(info="Epoch {}, train, loss={:.5f}".format(epoch, loss_value))
        running_loss.append(loss_value)

    return s.mean(running_loss), s.mean(running_mae), \
           s.mean(running_mse), s.mean(running_ssim)


def valid(epoch, loader, loss_fn, model, mse, ssim):
    model.eval()
    torch.no_grad()

    loop = prog(loader)

    running_loss = []
    running_mae = []
    running_mse = []
    running_ssim = []

    for batch_index, (data, target) in enumerate(loop):
        data = data.to(device)

        data[data < 0] = 0
        target[target < 0] = 0

        data = model(data)
        data[data < 0] = 0

        target = target.to(device).unsqueeze(1)

        with torch.no_grad():
            loss = loss_fn(data, target)

        loss_value = loss.item()

        running_mae.append(loss_value)
        running_mse.append(mse(data, target).item())
        running_ssim.append(ssim(data, target).item())

        mse.reset()
        ssim.reset()

        loop.set_postfix(info="Epoch {}, valid, loss={:.5f}".format(epoch, loss_value))
        running_loss.append(loss_value)

    return s.mean(running_loss), s.mean(running_mae), \
           s.mean(running_mse), s.mean(running_ssim)


def run(num_epochs, lr, epoch_to_start_from):
    torch.cuda.empty_cache()

    model = UNET(in_channels=4, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=5, verbose=True)

    torch_mse = MeanSquaredError().to(device)
    torch_ssim = StructuralSimilarityIndexMeasure().to(device)

    epochs_done = 0

    overall_training_loss = []
    overall_validation_loss = []

    overall_training_mae = []
    overall_validation_mae = []
    overall_training_mse = []
    overall_validation_mse = []
    overall_training_ssim = []
    overall_validation_ssim = []

    path = "{}_{}_{}_{}_{}/".format(
        "results",
        str(loss_fn.__class__.__name__),
        str(optimizer.__class__.__name__),
        str(UNET.__qualname__),
        lr
    )

    if not os.path.isdir(path):
        os.mkdir(path)

    if os.path.isfile(path + "model_epoch" + str(epoch_to_start_from) + ".pt") and epoch_to_start_from > 0:
        checkpoint = torch.load(path + "model_epoch" + str(epoch_to_start_from) + ".pt", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs_done = checkpoint['epoch']
        overall_training_loss = checkpoint['training_losses']
        overall_validation_loss = checkpoint['validation_losses']
        overall_training_mae = checkpoint['training_maes']
        overall_training_mse = checkpoint['training_mses']
        overall_training_ssim = checkpoint['training_ssims']
        overall_validation_mae = checkpoint['validation_maes']
        overall_validation_mse = checkpoint['validation_mses']
        overall_validation_ssim = checkpoint['validation_ssims']
        early_stopping = checkpoint['early_stopping']
    else:
        if epoch_to_start_from == 0:
            model.to(device)
        else:
            raise Exception("No model_epoch" + str(epoch_to_start_from) + ".pt found")

    model.to(device)

    train_loader = get_loader(path_train, batch_size, 1, num_workers, pin_memory)
    validation_loader = get_loader(path_validation, batch_size, 1, num_workers, pin_memory)

    for epoch in range(epochs_done + 1, num_epochs + 1):
        training_loss, training_mae, training_mse, training_ssim = train(
            epoch,
            train_loader,
            loss_fn,
            optimizer,
            scaler,
            model,
            torch_mse,
            torch_ssim
        )

        validation_loss, validation_mae, validation_mse, validation_ssim = valid(
            epoch,
            validation_loader,
            loss_fn, model,
            torch_mse,
            torch_ssim
        )

        overall_training_loss.append(training_loss)
        overall_validation_loss.append(validation_loss)

        overall_training_mae.append(training_mae)
        overall_validation_mae.append(validation_mae)

        overall_training_mse.append(training_mse)
        overall_validation_mse.append(validation_mse)

        overall_training_ssim.append(training_ssim)
        overall_validation_ssim.append(validation_ssim)

        early_stopping(validation_loss, model)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.cpu().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_losses': overall_training_loss,
            'validation_losses': overall_validation_loss,
            'training_maes': overall_training_mae,
            'training_mses': overall_training_mse,
            'training_ssims': overall_training_ssim,
            'validation_maes': overall_validation_mae,
            'validation_mses': overall_validation_mse,
            'validation_ssims': overall_validation_ssim,
            'early_stopping': early_stopping
        }, path + "model_epoch" + str(epoch) + ".pt")

        model.to(device)

        metrics = np.array([
            overall_training_loss,
            overall_validation_loss,
            overall_training_mae,
            overall_training_mse,
            overall_training_ssim,
            overall_validation_mae,
            overall_validation_mse,
            overall_validation_ssim,
        ], dtype='object')

        savetxt(path + "metrics.csv", metrics, delimiter=',',
                header="tloss,vloss,tmae,tmse,tssim,vmae,vmse,vssim", fmt='%s')

        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == '__main__':
    run(num_epochs=100, lr=1e-05, epoch_to_start_from=0)
