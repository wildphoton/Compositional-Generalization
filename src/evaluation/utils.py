#!/usr/bin/env python
"""
Created by zhenlinx on 02/25/2022
"""
import torch
from tqdm import tqdm
def infer(model, dataloader, mode):
    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device

        latents, targets = [], []
        for x, t in dataloader:
            x = x.to(device=device)
            z = model.embed(x, mode)

            latents.append(z.cpu())
            targets.append(t)

    latents = torch.cat(latents)
    targets = torch.cat(targets)

    return latents, targets