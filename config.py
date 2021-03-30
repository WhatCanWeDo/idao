from pathlib import Path
import torch
DATA_MODES = ['train', 'test', 'val']
TEST_DIR_PUBLIC = Path("./tests/public_test/")
TEST_DIR_PRIVATE = Path("./tests/private_test/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_ENCODER_NAME = "./model/label_encoder_27_03.pckl"
WEIGHTS_NAME = "./model/best_weights_27_03.pckl"
ARCHITECTURE_NAME = "./model/arch_27_03.pckl"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
