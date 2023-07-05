"""
Training module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from typing import Dict
from config import config


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# Transformation pipeline
TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])


class Net(nn.Module):
    """
    Neural network architecture for reading handwritten characters.
    """
    def __init__(self, num_output):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_output)

    def forward(self, x):
        """
        Perform a forward pass.
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def config_settings():
    """
    Configure torch settings prior to training.
    """
    # Remove any randomization from training so results are reproducible
    torch.backends.cudnn.enabled = False
    torch.manual_seed(config['RANDOM_SEED'])


def load_data(train_batch_size: int, val_batch_size: int, test_batch_size: int):
    """
    Return PyTorch data loaders in the form of (TRAIN, VAL, TEST).

    :param train_batch_size: Batch size of training data.
    :param val_batch_size: Batch size of validation data.
    :param test_batch_size: Batch size of testing data.
    :return: PyTorch data loaders in the form of (TRAIN, VAL, TEST).

    For information on data loaders, see https://pytorch.org/docs/stable/data.html.
    """
    # Split train set into (TRAIN, VAL) and get data loaders for these new datasets
    train_set = datasets.MNIST(root=os.path.join(os.path.dirname(__file__), config['RELATIVE_DATA_LOC']),
                               train=True, download=True, transform=TRANSFORM)
    train_subset, val_subset = torch.utils.data.random_split(train_set, [50000, 10000])
    train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=train_batch_size)
    val_loader = DataLoader(dataset=val_subset, shuffle=False, batch_size=val_batch_size)

    # Get test data loader
    test_loader = DataLoader(
        datasets.MNIST(
            root=os.path.join(os.path.dirname(__file__), config['RELATIVE_DATA_LOC']),
            train=False, download=True, transform=TRANSFORM
        ),
        batch_size=test_batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


def display_data_batch(train_data_loader) -> None:
    """
    Display a batch of the training data.

    :param train_data_loader: PyTorch DataLoader for training data.
    :return: None
    """
    num_display = 6
    train_data_iter = iter(train_data_loader)
    sample_images, sample_labels = next(train_data_iter)

    for i in range(num_display):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(sample_images[i][0], cmap='gray', interpolation='none')
        plt.title(f'Label: {sample_labels[i]}')
        plt.xticks([])
        plt.yticks([])

    plt.show(block=True)


def train_numeric(train_data_loader, val_data_loader, cache=None):
    """
    Train a numeric recognition model.

    :param train_data_loader: PyTorch DataLoader containing training data.
    :param val_data_loader: PyTorch DataLoader containing validation data.
    :param cache: Optional parameter specifying file paths for already trained
                  models. Default value is None, meaning create a model from
                  scratch. To specify a value for this parameter, pass in
                  a dictionary of the form:
                  {
                    'model': YOUR_RELATIVE_MODEL_PATH,
                    'optimizer': YOUR_RELATIVE_OPTIMIZER_PATH
                  }

    :return: Statistics from training. Specifically, returns a dictionary of the form:
             {
                'model': The trained model itself.
                'train_loss': A list containing the training loss at every LOG_INTERVAL
                              (this value is cleared every LOG_INTERVAL).
                'validation_accuracy': A list containing the validation accuracy at every
                                       LOG_INTERVAL.
                'seen_examples': A list containing the number of examples seen at every
                                 LOG_INTERVAL.
                'epoch_end_data': A dictionary of the form:
                                  {
                                    'epoch_end_seen_examples': Seen examples at end of
                                                               epoch.
                                    'epoch_end_train_loss': Training loss at end of
                                                            epoch.
                                    'epoch_end_validation_accuracy': Validation accuracy
                                                                     at end of epoch.
                                  }
             }

    Note: Model is saved at RELATIVE_MODEL_LOC/model_numeric.pth at every
          LOG_INTERVAL, and the optimizer is saved at
          RELATIVE_OPTIMIZER_LOC/optimizer_numeric.pth at every LOG_INTERVAL.
          Both the model and optimizer are also saved at every epoch, at the
          respective locations RELATIVE_MODEL_LOC/model_numeric_epoch_X.pth
          and RELATIVE_OPTIMIZER_LOC/optimizer_numeric_epoch_X.pth, where X
          represents the current epoch number.
    """
    model = Net(10)
    return _train(model, train_data_loader, val_data_loader, cache, 'NUMERIC')


def train_letter(train_data_loader, val_data_loader, cache=None):
    """
    Train a letter recognition model.

    :param train_data_loader: PyTorch DataLoader containing training data.
    :param val_data_loader: PyTorch DataLoader containing validation data.
    :param cache: Optional parameter specifying file paths for already trained
                  models. Default value is None, meaning create a model from
                  scratch. To specify a value for this parameter, pass in
                  a dictionary of the form:
                  {
                    'model': YOUR_RELATIVE_MODEL_PATH,
                    'optimizer': YOUR_RELATIVE_OPTIMIZER_PATH
                  }

    :return: Statistics from training. Specifically, returns a dictionary of the form:
             {
                'model': The trained model itself.
                'train_loss': A list containing the training loss at every LOG_INTERVAL
                              (this value is cleared every LOG_INTERVAL).
                'validation_accuracy': A list containing the validation accuracy at every
                                       LOG_INTERVAL.
                'seen_examples': A list containing the number of examples seen at every
                                 LOG_INTERVAL.
                'epoch_end_data': A dictionary of the form:
                                  {
                                    'epoch_end_seen_examples': Seen examples at end of
                                                               epoch.
                                    'epoch_end_train_loss': Training loss at end of
                                                            epoch.
                                    'epoch_end_validation_accuracy': Validation accuracy
                                                                     at end of epoch.
                                  }
             }

    Note: Model is saved at RELATIVE_MODEL_LOC/model_letter.pth at every
          LOG_INTERVAL, and the optimizer is saved at
          RELATIVE_OPTIMIZER_LOC/optimizer_letter.pth at every LOG_INTERVAL.
          Both the model and optimizer are also saved at every epoch, at the
          respective locations RELATIVE_MODEL_LOC/model_letter_epoch_X.pth
          and RELATIVE_OPTIMIZER_LOC/optimizer_numeric_letter_X.pth, where X
          represents the current epoch number.
    """
    model = Net(26)
    return _train(model, train_data_loader, val_data_loader, cache, 'LETTER')


def _train(model, train_data_loader, val_data_loader, cache, model_type):
    """
    Train a Net model.
    """
    # Create relative path locations without their file extensions
    relative_model_path_no_ext = f'{config["RELATIVE_MODEL_LOC"]}/model_{model_type.lower()}'
    relative_optimizer_path_no_ext = f'{config["RELATIVE_OPTIMIZER_LOC"]}/optimizer_{model_type.lower()}'

    # Define optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=config['ALPHA'], momentum=config['MOMENTUM'], weight_decay=1e-4)
    criterion = F.nll_loss

    # Load cached model if requested
    if cache:
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), cache['model'])))
        optimizer.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), cache['optimizer'])))

    # Tell PyTorch we're training the model
    model.train()

    # Define statistics data structures
    general_training_data = {
        'train_loss': [],
        'validation_accuracy': [],
        'seen_examples': []
    }
    epoch_end_data = {
        'epoch_end_seen_examples': [],
        'epoch_end_train_loss': [],
        'epoch_end_validation_accuracy': []
    }

    # Begin training
    for epoch in range(config['NUM_EPOCHS']):

        train_loss = 0.

        for batch_index, (batch_data, batch_targets) in enumerate(train_data_loader):
            # Zero out old gradients
            optimizer.zero_grad()

            # Make predictions and calculate loss
            output = model(batch_data)
            loss = criterion(output, batch_targets)

            # Perform a backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Update statistics
            train_loss += loss.item()

            if batch_index % config['LOG_INTERVAL'] == 0:
                # Evaluate model on validation data
                val_acc = evaluate(model, val_data_loader)

                # Store statistics
                general_training_data['train_loss'].append(loss.item())
                general_training_data['validation_accuracy'].append(val_acc)

                examples_seen_so_far = \
                    (batch_index * batch_targets.size(0)) + (epoch * len(train_data_loader.dataset))
                general_training_data['seen_examples'].append(examples_seen_so_far)

                # Switch back to training
                model.train()

                # Reset training loss
                train_loss = 0.

                # Update cached model and optimizer
                torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__),
                                                            f'{relative_model_path_no_ext}.pth'))
                torch.save(optimizer.state_dict(), os.path.join(os.path.dirname(__file__),
                                                                f'{relative_optimizer_path_no_ext}.pth'))

                print(f'Epoch {epoch} [{batch_index}]: {val_acc}')

        # Log the current training data end of this epoch.
        # This can be used to highlight the points on each training curve where epochs end.
        epoch_end_data['epoch_end_seen_examples'].append(general_training_data['seen_examples'][-1])
        epoch_end_data['epoch_end_train_loss'].append(general_training_data['train_loss'][-1])
        epoch_end_data['epoch_end_validation_accuracy'].append(general_training_data['validation_accuracy'][-1])

        # Save model and optimizer at the end of this epoch
        torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__),
                                                    f'{relative_model_path_no_ext}_epoch_{epoch}.pth'))
        torch.save(optimizer.state_dict(), os.path.join(os.path.dirname(__file__),
                                                        f'{relative_optimizer_path_no_ext}_epoch_{epoch}.pth'))

    return {
        'model': model,
        'train_loss': general_training_data['train_loss'],
        'validation_accuracy': general_training_data['validation_accuracy'],
        'seen_examples': general_training_data['seen_examples'],
        'epoch_end_data': epoch_end_data
    }


def evaluate(model, data_loader) -> float:
    """
    Evaluate the Neural Network using the accuracy metric.

    :param model: Neural Network to evaluate.
    :param data_loader: PyTorch DataLoader containing the data to evaluate
                        the model on.
    :return: Accuracy score achieved by the model, as a percentage.
    """
    model.eval()
    correct = 0

    # Disable gradient computation
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            # Run model
            output = model(batch_data)

            # Get predictions
            pred = output.argmax(dim=1, keepdim=True).reshape(-1)

            # Increment correct count
            correct += (pred == batch_labels).sum().item()

    return correct / len(data_loader.dataset)


def plot_results(results_dict: Dict) -> None:
    """
    Plot the training statistics as a function of the number of examples seen.

    :param results_dict: Return value from either of the train_numeric() or train_letter()
                         procedures.
    :return: None
    """
    # Extract statistics
    epoch_end_data = results_dict['epoch_end_data']

    seen_examples = results_dict['seen_examples']
    train_loss = results_dict['train_loss']
    validation_accuracy = results_dict['validation_accuracy']

    epoch_end_seen_examples = epoch_end_data['epoch_end_seen_examples']
    epoch_end_train_loss = epoch_end_data['epoch_end_train_loss']
    epoch_end_validation_accuracy = epoch_end_data['epoch_end_validation_accuracy']

    # Create figure with two subplots
    fig, ax = plt.subplots(nrows=1, ncols=2)

    # Plot training loss on leftmost subplot
    ax[0].plot(seen_examples, train_loss, label='Training Loss')
    ax[0].set_xlabel('Seen Examples')
    ax[0].set_ylabel('Training Loss')
    ax[0].plot(epoch_end_seen_examples, epoch_end_train_loss, linestyle="", marker='o', color='r',
               label='Training Loss at Epoch End')
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.115), ncol=1)

    # Plot validation accuracy on rightmost subplot
    ax[1].plot(seen_examples, validation_accuracy, label='Validation Accuracy')
    ax[1].set_xlabel('Seen Examples')
    ax[1].set_ylabel('Accuracy')
    ax[1].plot(epoch_end_seen_examples, epoch_end_validation_accuracy, linestyle="", marker='o', color='r',
               label='Validation Accuracy at Epoch End')
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.115), ncol=1)

    # Set figure size
    fig.set_size_inches(10, 7)

    # Show the plot
    plt.show(block=True)


def test_final_model(test_data_loader, relative_model_path, model_type) -> float:
    """
    Evaluate the final model on the test set.

    :param test_data_loader: PyTorch DataLoader containing the test data.
    :param relative_model_path: Relative path to the trained CNN to evaluate.
    :param model_type: String representing the type of model. Must be one of
                       'NUMERIC' or 'LETTER'.
    """
    # Load model
    if model_type == 'NUMERIC':
        model = Net(10)
    else:
        model = Net(26)

    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), relative_model_path)))

    # Evaluate model
    return evaluate(model, test_data_loader)


if __name__ == '__main__':
    config_settings()
    train_data_loader, val_data_loader, test_data_loader = \
        load_data(config['TRAIN_BATCH_SIZE'], config['VALIDATION_BATCH_SIZE'], config['TEST_BATCH_SIZE'])

    # Train Code (Numeric):
    # cache = {
    #     'model': f'{RELATIVE_MODEL_LOC}/model_numeric.pth',
    #     'optimizer': f'{RELATIVE_OPTIMIZER_LOC}/optimizer_numeric.pth'
    # }
    cache = None

    results_dict = train_numeric(train_data_loader, val_data_loader, cache=cache)
    plot_results(results_dict)


    # Test Code:
    # print(test_final_model(test_data_loader, f'{RELATIVE_MODEL_LOC}/model_numeric.pth', 'NUMERIC'))
    