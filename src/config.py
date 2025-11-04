# Project configuration
class Config:
    # Image settings
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_CHANNELS = 3
    
    # Data settings
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Training settings
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    
    # Paths
    DATA_PATH = "data/raw"
    PROCESSED_PATH = "data/processed"