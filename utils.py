import pandas as pd
import numpy as np
import torch

def get_predictions(loader, model, device):
    model.eval()

    saved_preds = []
    true_labels = []

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            socres = model(x)

            saved_preds += socres.tolist()
            true_labels += y.tolist()

    model.train()
    return saved_preds, true_labels