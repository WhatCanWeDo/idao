import pickle

import numpy as np
import pandas as pd
import torch
from model import IDAONet
from torch.utils.data import DataLoader

from config import TEST_DIR_PUBLIC, TEST_DIR_PRIVATE, BATCH_SIZE, WEIGHTS_NAME, LABEL_ENCODER_NAME, ARCHITECTURE_NAME, device
from dataset import IDAODataset
from utils import predict, round_nearest

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_files = sorted(list(TEST_DIR_PUBLIC.rglob('*.png')) + list(TEST_DIR_PRIVATE.rglob("*.png")))
    test_set = IDAODataset(test_files, mode='test')
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    submit = pd.DataFrame(columns=['id'])
    submit['id'] = [path.name.replace('.png', '') for path in test_set.files]

    model = pickle.load(open(ARCHITECTURE_NAME, "rb")).to(device)
    model.load_state_dict(pickle.load(open(WEIGHTS_NAME, 'rb')))
    probs = predict(model, test_loader, device)

    label_encoder = pickle.load(open(LABEL_ENCODER_NAME, 'rb'))

    preds_cl = label_encoder.inverse_transform(np.round(probs['classification']).astype(int))
    convert_to_int = {
        'ER': 1,
        'NR': 0,
    }
    submit['classification_predictions'] = [convert_to_int[i] for i in preds_cl]

    preds_reg = round_nearest(list(probs['regression']))
    submit['regression_predictions'] = preds_reg
    submit.to_csv('submission.csv', index=False)
