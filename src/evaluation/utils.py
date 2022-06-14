#!/usr/bin/env python

import torch
from copy import deepcopy

def infer(model, dataloader, mode):
    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device

        latents, targets = [], []
        for x, t in dataloader:
            x = x.to(device=device)
            z = model.embed(x, mode)
            latents.append(z.cpu())
            targets.append(deepcopy(t))

    latents = torch.cat(latents)
    targets = torch.cat(targets)

    return latents, targets