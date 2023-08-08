import warnings
warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')
from src.train import main as train

if __name__ == "__main__":
    train()