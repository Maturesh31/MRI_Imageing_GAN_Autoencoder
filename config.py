import os
import torch

class Config:
    # Random seed for reproducibility
    SEED = 42

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "database", "ABIDE", "Combined Data")
    CSV1_PATH = os.path.join(BASE_DIR, "database", "ABIDE", "abide1_data.csv")
    CSV2_PATH = os.path.join(BASE_DIR, "database", "ABIDE", "abide2_data.csv")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    SPLIT_FILE = os.path.join(BASE_DIR, "split_indices.pkl")

    # Data specifications
    IMAGE_SIZE = (224, 224)
    CHANNELS = 1  # 2D Grayscale sMRI slice
    TEST_SPLIT_RATIO = 0.20  # 80% train, 20% test stratified

    # Training Hyperparameters for CNN (consistent across E0-E3)
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 0  # 0 is recommended for PyTorch multiprocessing on Windows
    CACHE_IN_MEMORY = True  # Cache slices in RAM for maximum speed

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(Config.RESULTS_DIR, exist_ok=True)
