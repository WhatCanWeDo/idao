from pathlib import Path
import torch
DATA_MODES = ['train', 'test', 'val']
TEST_DIR_PUBLIC = Path("./tests/public_test/")
TEST_DIR_PRIVATE = Path("./tests/private_test/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_ENCODER_NAME = "./model/label_encoder.pckl"
ARCHITECTURE_NAME = "./model/model_dump"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
