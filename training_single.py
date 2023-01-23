import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm as prog
from numpy import savetxt
import statistics as s
import numpy as np
import matplotlib.pyplot as plt

from builder.dataset_provider_single import (
    get_loader
)

from pytorchtools import EarlyStopping

from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall

from torchmetrics.regression import MeanSquaredError
from torchmetrics.image import StructuralSimilarityIndexMeasure

import shutup

shutup.please()

from model.model_single_shape import UNET_SHAPE
from model.model_single_height import UNET_HEIGHT

from configuration import (
    device,
    batch_size,
    num_workers,
    pin_memory
)


def train(epoch, loader, loss_fn_shape, loss_fn_height, optimizer_shape, optimizer_height, scaler, model_shape, model_height, accuracy, f1, recall, precision, mse, ssim):
    torch.enable_grad()

    model_shape.train()
    model_height.train()

    loop = prog(loader)

    running_loss_shape = []
    running_loss_height = []

    running_mse = []
    running_ssim = []

    running_accuracy = []
    running_f1 = []
    running_recall = []
    running_precision = []

    for batch_index, (_, data, target_shape, target_height) in enumerate(loop):
        optimizer_shape.zero_grad(set_to_none=True)
        optimizer_height.zero_grad(set_to_none=True)
        data = data.to(device)

        target_shape = target_shape.to(device).unsqueeze(1)
        target_height = target_height.to(device).unsqueeze(1)

        target_shape = torch.round(target_shape)

        with torch.cuda.amp.autocast():
            prediction_shape = model_shape(data)
            prediction_shape = torch.nan_to_num(prediction_shape, nan=0.0)

            prediction_height = model_height(data, prediction_shape.detach().clone())

            loss_shape = loss_fn_shape(prediction_shape, target_shape)
            loss_height = loss_fn_height(prediction_height, target_height)

        scaler.scale(loss_shape).backward(retain_graph=True)
        scaler.scale(loss_height).backward()

        scaler.unscale_(optimizer_shape)

        scaler.step(optimizer_shape)
        scaler.step(optimizer_height)

        scaler.update()

        loss_value_shape = loss_shape.item()
        loss_value_height = loss_height.item()

        running_accuracy.append(accuracy(prediction_shape, target_shape).item())
        running_f1.append(f1(prediction_shape, target_shape).item())
        running_recall.append(recall(prediction_shape, target_shape).item())
        running_precision.append(precision(prediction_shape, target_shape).item())

        prediction_height = prediction_height.type(torch.DoubleTensor)
        target_height = target_height.type(torch.DoubleTensor)

        running_mse.append(mse(prediction_height, target_height).item())
        running_ssim.append(ssim(prediction_height, target_height).item())

        accuracy.reset()
        f1.reset()
        recall.reset()
        precision.reset()

        mse.reset()
        ssim.reset()

        loop.set_postfix(info="Epoch {}, train, loss_shape={:.4f}, loss_height={:.4f}".format(epoch, loss_value_shape, loss_value_height))
        running_loss_shape.append(loss_value_shape)
        running_loss_height.append(loss_value_height)

    return s.mean(running_loss_shape), s.mean(running_loss_height), s.mean(running_accuracy), \
           s.mean(running_f1), s.mean(running_recall), s.mean(running_precision), \
           s.mean(running_mse), s.mean(running_ssim)


def valid(epoch, loader, loss_fn_shape, loss_fn_height, model_shape, model_height, accuracy, f1, recall, precision, mse, ssim):
    model_shape.eval()
    model_height.eval()

    loop = prog(loader)

    running_loss_shape = []
    running_loss_height = []

    running_mse = []
    running_ssim = []

    running_accuracy = []
    running_f1 = []
    running_recall = []
    running_precision = []

    for batch_index, (_, data, target_shape, target_height) in enumerate(loop):
        data = data.to(device)

        target_shape = target_shape.to(device).unsqueeze(1)
        target_height = target_height.to(device).unsqueeze(1)

        target_shape = torch.round(target_shape)

        with torch.no_grad():
            prediction_shape = model_shape(data)
            prediction_height = model_height(data, prediction_shape.detach().clone())

            loss_shape = loss_fn_shape(prediction_shape, target_shape)
            loss_height = loss_fn_height(prediction_height, target_height)

        loss_value_shape = loss_shape.item()
        loss_value_height = loss_height.item()

        running_accuracy.append(accuracy(prediction_shape, target_shape).item())
        running_f1.append(f1(prediction_shape, target_shape).item())
        running_recall.append(recall(prediction_shape, target_shape).item())
        running_precision.append(precision(prediction_shape, target_shape).item())

        running_mse.append(mse(prediction_height, target_height).item())
        running_ssim.append(ssim(prediction_height, target_height).item())

        accuracy.reset()
        f1.reset()
        recall.reset()
        precision.reset()

        mse.reset()
        ssim.reset()

        loop.set_postfix(info="Epoch {}, train, loss_shape={:.4f}, loss_height={:.4f}".format(epoch, loss_value_shape, loss_value_height))
        running_loss_shape.append(loss_value_shape)
        running_loss_height.append(loss_value_height)

    return s.mean(running_loss_shape), s.mean(running_loss_height), s.mean(running_accuracy), \
           s.mean(running_f1), s.mean(running_recall), s.mean(running_precision), \
           s.mean(running_mse), s.mean(running_ssim)


def run(num_epochs, lr_shape, lr_height, epoch_to_start_from, below_meters_equal_to_0):
    torch.cuda.empty_cache()

    model_shape = UNET_SHAPE(in_channels=4, out_channels=1).to(device)
    model_height = UNET_HEIGHT(in_channels=4, out_channels=1).to(device)
    optimizer_shape = optim.Adam(model_shape.parameters(), lr=lr_shape, weight_decay=1e-04)
    optimizer_height = optim.Adam(model_height.parameters(), lr=lr_height, weight_decay=1e-04)
    loss_fn_shape = nn.BCEWithLogitsLoss()
    loss_fn_height = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()
    early_stopping_shape = EarlyStopping(patience=5, verbose=True)
    early_stopping_height = EarlyStopping(patience=5, verbose=True)

    torch_mse = MeanSquaredError().to(device)
    torch_ssim = StructuralSimilarityIndexMeasure().to(device)

    torch_accuracy = BinaryAccuracy().to(device)
    torch_f1 = BinaryF1Score().to(device)
    torch_recall = BinaryRecall().to(device)
    torch_precision = BinaryPrecision().to(device)

    epochs_done = 0

    overall_training_loss_shape = []
    overall_training_loss_height = []
    overall_validation_loss_shape = []
    overall_validation_loss_height = []

    overall_training_mse = []
    overall_training_ssim = []
    overall_training_accuracy = []
    overall_training_f1 = []
    overall_training_recall = []
    overall_training_precision = []

    overall_validation_mse = []
    overall_validation_ssim = []
    overall_validation_accuracy = []
    overall_validation_f1 = []
    overall_validation_recall = []
    overall_validation_precision = []

    path = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}/".format(
        "results_single",
        str(loss_fn_shape.__class__.__name__),
        str(loss_fn_height.__class__.__name__),
        str(optimizer_shape.__class__.__name__),
        str(optimizer_height.__class__.__name__),
        str(UNET_SHAPE.__qualname__),
        str(UNET_HEIGHT.__qualname__),
        lr_shape,
        lr_height,
        "below" + str(below_meters_equal_to_0)
    )

    if not os.path.isdir(path):
        os.mkdir(path)

    if os.path.isfile(path + "model_epoch" + str(epoch_to_start_from) + ".pt") and epoch_to_start_from > 0:
        checkpoint = torch.load(path + "model_epoch" + str(epoch_to_start_from) + ".pt", map_location='cpu')
        model_shape.load_state_dict(checkpoint['model_state_dict_shape'])
        model_height.load_state_dict(checkpoint['model_state_dict_height'])
        optimizer_shape.load_state_dict(checkpoint['optimizer_state_dict_shape'])
        optimizer_height.load_state_dict(checkpoint['optimizer_state_dict_height'])
        epochs_done = checkpoint['epoch']

        overall_training_loss_shape = checkpoint['training_losses_shape']
        overall_training_loss_height = checkpoint['training_losses_height']
        overall_validation_loss_shape = checkpoint['validation_losses_shape']
        overall_validation_loss_height = checkpoint['validation_losses_height']

        overall_training_mse = checkpoint['training_mses']
        overall_training_ssim = checkpoint['training_ssims']
        overall_training_accuracy = checkpoint['training_accuracies']
        overall_training_f1 = checkpoint['training_f1s']
        overall_training_recall = checkpoint['training_recalls']
        overall_training_precision = checkpoint['training_precisions']

        overall_validation_mse = checkpoint['validation_mses']
        overall_validation_ssim = checkpoint['validation_ssims']
        overall_validation_accuracy = checkpoint['validation_accuracies']
        overall_validation_f1 = checkpoint['validation_f1s']
        overall_validation_recall = checkpoint['validation_recalls']
        overall_validation_precision = checkpoint['validation_precisions']

        early_stopping_shape = checkpoint['early_stopping_shape']
        early_stopping_height = checkpoint['early_stopping_height']
    else:
        if epoch_to_start_from == 0:
            model_shape.to(device)
            model_height.to(device)
        else:
            raise Exception("No model_epoch" + str(epoch_to_start_from) + ".pt found")

    model_shape.to(device)
    model_height.to(device)

    train_loader = get_loader(
        "output/single_belows/train",
        batch_size,
        percentage_load=1,
        below_m=below_meters_equal_to_0,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    validation_loader = get_loader(
        "output/single_belows/validation",
        batch_size,
        percentage_load=1,
        below_m=below_meters_equal_to_0,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    for epoch in range(epochs_done + 1, num_epochs + 1):
        tr_loss_shape, tr_loss_height, tr_accuracy, tr_f1, tr_recall, tr_precision, tr_mse, tr_ssim = train(
            epoch=epoch,
            loader=train_loader,
            loss_fn_shape=loss_fn_shape,
            loss_fn_height=loss_fn_height,
            optimizer_shape=optimizer_shape,
            optimizer_height=optimizer_height,
            scaler=scaler,
            model_shape=model_shape,
            model_height=model_height,
            accuracy=torch_accuracy,
            f1=torch_f1,
            recall=torch_recall,
            precision=torch_precision,
            mse=torch_mse,
            ssim=torch_ssim
        )

        val_loss_shape, val_loss_height, val_accuracy, val_f1, val_recall, val_precision, val_mse, val_ssim = valid(
            epoch=epoch,
            loader=validation_loader,
            loss_fn_shape=loss_fn_shape,
            loss_fn_height=loss_fn_height,
            model_shape=model_shape,
            model_height=model_height,
            accuracy=torch_accuracy,
            f1=torch_f1,
            recall=torch_recall,
            precision=torch_precision,
            mse=torch_mse,
            ssim=torch_ssim
        )

        overall_training_loss_shape.append(tr_loss_shape)
        overall_training_loss_height.append(tr_loss_height)
        overall_validation_loss_shape.append(val_loss_shape)
        overall_validation_loss_height.append(val_loss_height)

        overall_training_f1.append(tr_f1)
        overall_training_accuracy.append(tr_accuracy)
        overall_training_recall.append(tr_recall)
        overall_training_precision.append(tr_precision)
        overall_training_mse.append(tr_mse)
        overall_training_ssim.append(tr_ssim)

        overall_validation_f1.append(val_f1)
        overall_validation_accuracy.append(val_accuracy)
        overall_validation_recall.append(val_recall)
        overall_validation_precision.append(val_precision)
        overall_validation_mse.append(val_mse)
        overall_validation_ssim.append(val_ssim)

        early_stopping_shape(val_loss_shape, model_shape)
        early_stopping_height(val_loss_height, model_height)

        torch.save({
            'epoch': epoch,
            'model_state_dict_shape': model_shape.cpu().state_dict(),
            'optimizer_state_dict_shape': optimizer_shape.state_dict(),
            'model_state_dict_height': model_height.cpu().state_dict(),
            'optimizer_state_dict_height': optimizer_height.state_dict(),
            'training_losses_shape': overall_training_loss_shape,
            'training_losses_height': overall_training_loss_height,
            'validation_losses_shape': overall_validation_loss_shape,
            'validation_losses_height': overall_validation_loss_height,

            'training_accuracies': overall_training_accuracy,
            'training_f1s': overall_training_f1,
            'training_recalls': overall_training_recall,
            'training_precisions': overall_training_precision,
            'training_mses': overall_training_mse,
            'training_ssims': overall_training_ssim,

            'validation_accuracies': overall_validation_accuracy,
            'validation_f1s': overall_validation_f1,
            'validation_recalls': overall_validation_recall,
            'validation_precisions': overall_validation_precision,
            'validation_mses': overall_validation_mse,
            'validation_ssims': overall_validation_ssim,

            'early_stopping_shape': early_stopping_shape,
            'early_stopping_height': early_stopping_height
        }, path + "model_epoch" + str(epoch) + ".pt")

        model_shape.to(device)
        model_height.to(device)

        metrics = np.array([
            overall_training_loss_shape,
            overall_training_loss_height,
            overall_validation_loss_shape,
            overall_validation_loss_height,

            overall_training_accuracy,
            overall_training_f1,
            overall_training_recall,
            overall_training_precision,
            overall_training_mse,
            overall_training_ssim,

            overall_validation_accuracy,
            overall_validation_f1,
            overall_validation_recall,
            overall_validation_precision,
            overall_validation_mse,
            overall_validation_ssim
        ], dtype='object')

        savetxt(path + "metrics.csv", metrics, delimiter=',',
                header="tloss_shape,tloss_height,vloss_shape,vloss_height,tacc,tf1,trec,tprec,tmse,tssim,vacc,vf1,vrec,vprec,vmse,vssim",
                fmt='%s')

        if early_stopping_shape.early_stop or early_stopping_height.early_stop:
            print("Early stopping")
            break


if __name__ == '__main__':
    run(num_epochs=100, lr_shape=1e-05, lr_height=1e-05, epoch_to_start_from=0, below_meters_equal_to_0=5)
