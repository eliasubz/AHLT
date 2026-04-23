#! /usr/bin/python3

import sys, os
import random

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torchinfo import summary

from dataset import *
from codemaps import *

from network import nercLSTM, criterion, DeepNercLSTM, WideNercLSTM, FlexibleNercLSTM
from torch.utils.data import Subset

import numpy as np
from sklearn.metrics import f1_score

random.seed(2345)
torch.manual_seed(2345)
torch.cuda.manual_seed(2345)
torch.backends.cudnn.deterministic = True

# use gpu if available
used_device = "cuda:0" if torch.cuda.is_available() else "cpu"


def _as_float(v, default=None):
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except Exception:
        return default


def _as_int(v, default=None):
    if v is None:
        return default
    if isinstance(v, int):
        return v
    try:
        return int(str(v))
    except Exception:
        return default


def _build_loss(codes, traindata, params):
    """Build a CrossEntropyLoss aligned with our padded label tensors.

    - Always ignores PAD (index 0).
    - Optional class weighting to help rare labels (esp. drug_n).
    """
    pad_idx = 0

    weighting = str(params.get("class_weighting", "none")).lower()
    label_smoothing = _as_float(params.get("label_smoothing"), default=0.0) or 0.0
    if weighting in ("none", "0", "false", "off"):
        return nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=label_smoothing,
        )

    # Count label frequencies from the *training* gold labels.
    n_labels = codes.get_n_labels()
    counts = np.zeros(n_labels, dtype=np.int64)
    for _, _, labels in traindata.sentences():
        for lab in labels:
            idx = codes.label2idx(lab)
            if idx != pad_idx:
                counts[idx] += 1

    # Avoid division by zero for labels not present.
    counts[counts == 0] = 1

    # Weight schemes (all ignore PAD anyway)
    if weighting in ("inv", "inverse", "inverse_freq"):
        weights = 1.0 / counts.astype(np.float32)
    elif weighting in ("sqrt_inv", "inverse_sqrt"):
        weights = 1.0 / np.sqrt(counts.astype(np.float32))
    else:
        # Unknown scheme -> behave like none
        return nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=label_smoothing,
        )

    # Normalize to keep loss scale roughly stable.
    weights = weights / weights.mean()
    weights[pad_idx] = 0.0

    # Optional targeted boost for drug_n BIO labels.
    boost = _as_float(params.get("drug_n_boost"), default=1.0)
    if boost and boost != 1.0:
        for prefix in ("B-", "I-"):
            key = f"{prefix}drug_n"
            if key in codes.label_index:
                weights[codes.label2idx(key)] *= float(boost)

    w = torch.tensor(weights, dtype=torch.float32)
    if used_device.startswith("cuda"):
        w = w.to(torch.device(used_device))
    return nn.CrossEntropyLoss(
        ignore_index=pad_idx,
        weight=w,
        label_smoothing=label_smoothing,
    )


# ----------------------------------------------
def train(network, epoch, train_loader, optimizer, clip_grad=None):
    network.to(torch.device(used_device))

    network.train()
    seen = 0
    acc_loss = 0
    for batch_idx, X in enumerate(train_loader):
        target = X.pop()
        optimizer.zero_grad()
        output = network(*X)
        output = output.flatten(0, 1)
        target = target.flatten(0, 1)
        loss = criterion(output, target)
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=clip_grad)
        optimizer.step()
        acc_loss += loss.item()
        avg_loss = acc_loss / (batch_idx + 1)
        seen += len(X[0])
        print(
            "Train Epoch {}: batch:{}/{} sentence:{}/{} [{:.2f}%] Loss:{:.6f}\r".format(
                epoch,
                batch_idx + 1,
                len(train_loader),
                seen,
                len(train_loader.dataset),
                100.0 * (batch_idx + 1) / len(train_loader),
                avg_loss,
            ),
            flush=True,
            end="",
        )
    print()


# ----------------------------------------------
def validation(network, val_loader):
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X in val_loader:
            target = X.pop()
            output = network(*X)
            output = output.flatten(0, 1)
            target = target.flatten(0, 1)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            total += target.size()[0]
    test_loss /= len(val_loader)
    acc = 100.0 * correct / total
    print(
        "Validation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            test_loss, correct, total, acc
        )
    )
    return acc


# ----------------------------------------------
def encode_dataset(ds, codes, params):
    X = codes.encode_words(ds)
    y = codes.encode_labels(ds)
    if used_device == "cuda:0":
        X = [x.to(torch.device(used_device)) for x in X]
        y = y.to(torch.device(used_device))
    return DataLoader(TensorDataset(*X, y), batch_size=params["batch_size"])


def get_f1_score(network, val_loader, codes):
    network.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X in val_loader:
            target = X.pop()
            output = network(*X)

            # Get predictions and move to CPU
            predictions = torch.argmax(output, dim=-1).cpu().numpy()
            target = target.cpu().numpy()

            # Flatten and filter out padding (index 0)
            mask = target != 0
            all_preds.extend(predictions[mask])
            all_targets.extend(target[mask])

    # Calculate Macro F1
    # We ignore label 0 (PAD) and usually label for 'O'
    # to see how we are doing on actual entities.
    # Find the index of 'O' in your codes to exclude it if you want the "Strict" F1
    idx_O = codes.label2idx("O") if "O" in codes.label_index else -1

    # labels list should be all label indices EXCEPT padding and 'O'
    entity_labels = [i for i in range(1, codes.get_n_labels()) if i != idx_O]

    macro_f1 = f1_score(
        all_targets, all_preds, labels=entity_labels, average="macro", zero_division=0
    )
    return macro_f1 * 100


# # ----------------------------------------------
# def do_train(trainfile, valfile, params, modelname):

#     # set default values if some parameter is missing
#     if "max_len" not in params:
#         params["max_len"] = 150
#     if "suf_len" not in params:
#         params["suf_len"] = 5
#     if "batch_size" not in params:
#         params["batch_size"] = 16
#     if "epochs" not in params:
#         params["epochs"] = 10

#     # load pickle datasets (or parse if needed)
#     traindata = Dataset(trainfile)
#     valdata = Dataset(valfile)

#     # create indexes from training data
#     codes = Codemaps(traindata, params)
#     # encode datasets
#     train_loader = encode_dataset(traindata, codes, params)
#     val_loader = encode_dataset(valdata, codes, params)

#     # build network
#     network = DeepNercLSTM(codes)
#     summary(network)

#     #  network = nercLSTM(codes)
#     #  summary(network)

#     #  network = WideNercLSTM(codes)
#     #  summary(network)

#     # save indexs
#     os.makedirs(modelname, exist_ok=True)
#     torch.save(network, os.path.join(modelname, "network.nn"))
#     codes.save(os.path.join(modelname, "codemaps"))
#     # train each epoch, keep the best model on validation
#     best = 0
#     for epoch in range(params["epochs"]):
#         train(network, epoch, train_loader)
#         acc = validation(network, val_loader)
#         if acc > best:
#             best = acc
#             torch.save(network, os.path.join(modelname, f"network.nn"))


def do_train(trainfile, valfile, params, modelname):
    # Set default values if some parameter is missing
    if "max_len" not in params:
        params["max_len"] = 150
    if "suf_len" not in params:
        params["suf_len"] = 5
    if "batch_size" not in params:
        params["batch_size"] = 16
    if "epochs" not in params:
        params["epochs"] = 5

    # Architecture params (fallback to your original defaults if not provided)
    emb_sizes = params.get("emb_sizes", (100, 100, 50))
    hidden_size = params.get("hidden_size", 200)
    num_layers = params.get("num_layers", 1)
    dropout = params.get("dropout", 0.1)

    # Load datasets
    traindata = Dataset(trainfile)
    valdata = Dataset(valfile)

    # Create indexes and encode
    codes = Codemaps(traindata, params)
    train_loader = encode_dataset(traindata, codes, params)
    val_loader = encode_dataset(valdata, codes, params)

    # Build loss aligned with our evaluation goals
    global criterion
    criterion = _build_loss(codes, traindata, params)

    # Optimizer knobs
    lr = _as_float(params.get("lr"), default=1e-3)
    weight_decay = _as_float(params.get("weight_decay"), default=1e-5)
    clip_grad = _as_float(params.get("clip_grad"), default=None)
    patience = _as_int(params.get("patience"), default=3)
    min_delta = _as_float(params.get("min_delta"), default=0.05)
    lr_factor = _as_float(params.get("lr_factor"), default=0.5)
    lr_patience = _as_int(params.get("lr_patience"), default=1)

    # Build network using the flexible variant
    network = FlexibleNercLSTM(
        codes,
        emb_sizes=emb_sizes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    # Ensure model is on the correct device before summary/training
    if "used_device" in globals() and used_device.startswith("cuda"):
        network.to(torch.device(used_device))

    # torchinfo.summary can be slow and is not needed for sweeps.
    # Enable explicitly via: summary=1
    if str(params.get("summary", "0")).lower() in ("1", "true", "yes", "on"):
        summary(network)

    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=lr_factor,
        patience=lr_patience,
    )

    # Save indexes and initial model (using .nn extension)
    os.makedirs(modelname, exist_ok=True)
    torch.save(network, os.path.join(modelname, "network.nn"))
    codes.save(os.path.join(modelname, "codemaps"))

    # Train each epoch, keep the best model on validation
    best = -1.0
    stale_epochs = 0
    for epoch in range(params["epochs"]):
        train(network, epoch, train_loader, optimizer, clip_grad=clip_grad)
        validation(network, val_loader)
        # Use F1 instead of Accuracy to decide "Best"
        current_f1 = get_f1_score(network, val_loader, codes)
        scheduler.step(current_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Validation macro-F1: {current_f1:.2f}% | lr={current_lr:.6g}")

        if current_f1 > best + min_delta:
            best = current_f1
            stale_epochs = 0
            torch.save(network, os.path.join(modelname, "network.nn"))
            print(f"New best model saved! F1: {best:.2f}%")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no macro-F1 improvement > {min_delta:.2f} for {patience} epoch(s))."
                )
                break

    return best


## --------- MAIN PROGRAM -----------
## --
## -- Usage:  train.py train.pck devel.pck modelname [batch_size=N] [max_len=N] [suf_len=N]
## --

if __name__ == "__main__":
    # files to process
    trainfile = sys.argv[1]
    validationfile = sys.argv[2]
    modelname = sys.argv[3]

    params = {}
    for p in sys.argv[4:]:
        k, v = p.split("=")
        params[k] = int(v)

    do_train(trainfile, validationfile, params, modelname)
