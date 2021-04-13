import numpy as np
import torch


def round_nearest(a):
    nums = [1, 3, 6, 10, 20, 30]
    for i in range(len(a)):
        _, num = min((abs(num - a[i]), num) for (idx, num) in enumerate(nums))
        a[i] = num
    return a


def predict(model, test_loader, device):
    with torch.no_grad():
        res = {'classification': np.empty(shape=0),
               'regression': np.empty(shape=0)}

        for inputs in test_loader:
            inputs = inputs.to(device)
            model.eval()
            outputs_cl, outputs_reg = model(inputs)
            outputs_cl = np.array(outputs_cl.cpu().squeeze())
            outputs_reg = np.array(outputs_reg.cpu().squeeze())
            res['classification'] = np.concatenate((res['classification'], outputs_cl), axis=None)
            res['regression'] = np.concatenate((res['regression'], outputs_reg), axis=None)
    return res
