config = {
    # Hyperparameters
    'TRAIN_BATCH_SIZE': 32,
    'VALIDATION_BATCH_SIZE': 32,
    'TEST_BATCH_SIZE': 1000,
    'NUM_EPOCHS': 2,
    'ALPHA': 0.01,
    'MOMENTUM': 0.5,

    # Statistics Logging & Model Storing Interval
    'LOG_INTERVAL': 10,

    # Relative Storage Directory Paths
    'RELATIVE_MODEL_LOC': 'results',
    'RELATIVE_OPTIMIZER_LOC': 'results',
    'RELATIVE_DATA_LOC': 'data',

    # Miscellaneous
    'RANDOM_SEED': 1,
}
