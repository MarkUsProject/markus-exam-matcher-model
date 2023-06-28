"""
Training module.
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

TRAIN_BATCH_SIZE = 32
NUM_EPOCHS = 1
TEST_BATCH_SIZE = 1000
ALPHA = 0.01
MOMENTUM = 0.5
LOG_INTERVAL = 10
RANDOM_SEED = 1

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

RELATIVE_MODEL_LOC = 'results/model.pth'
RELATIVE_OPTIMIZER_LOC = 'results/optimizer.pth'
RELATIVE_DATA_LOC = 'data'

# Transformation pipeline
TRANSFORM = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
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
    """Configure torch settings prior to training."""
    # Remove any randomization from training so results are reproducible
    torch.backends.cudnn.enabled = False
    torch.manual_seed(RANDOM_SEED)


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
    train_set = datasets.MNIST(root=os.path.join(os.path.dirname(__file__), RELATIVE_DATA_LOC),
                               train=True, download=True, transform=TRANSFORM)
    train_subset, val_subset = torch.utils.data.random_split(train_set, [50000, 10000])
    train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=train_batch_size)
    val_loader = DataLoader(dataset=val_subset, shuffle=False, batch_size=val_batch_size)

    # Get test data loader
    test_loader = DataLoader(
        datasets.MNIST(
            root=os.path.join(os.path.dirname(__file__), RELATIVE_DATA_LOC),
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


def train_numeric(train_data_loader, val_data_loader, cache=False):
    """Train a numeric recognition model."""
    model = Net(10)
    return _train(model, train_data_loader, val_data_loader, cache)


def train_letter(train_data_loader, val_data_loader, cache=False):
    """Train a letter recognition model."""
    model = Net(26)
    return _train(model, train_data_loader, val_data_loader, cache)


def _train(model, train_data_loader, val_data_loader, cache):
    """Train a Net model."""
    # Define optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=ALPHA, momentum=MOMENTUM, weight_decay=1e-4)
    criterion = F.nll_loss

    # Load cached model if requested
    if cache:
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), RELATIVE_MODEL_LOC)))
        optimizer.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), RELATIVE_OPTIMIZER_LOC)))

    # Tell PyTorch we're training the model
    model.train()

    # Define statistics data structures
    train_loss_lst = []
    val_accuracy_lst = []
    seen_examples = []

    # Begin training
    for epoch in range(NUM_EPOCHS):

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

            if batch_index % LOG_INTERVAL == 0:
                # Evaluate model on validation data
                val_acc = evaluate(model, val_data_loader)

                # Store statistics
                train_loss_lst.append(loss.item())
                val_accuracy_lst.append(val_acc)

                examples_seen_so_far = \
                    (batch_index * batch_targets.size(0)) + (epoch * len(train_data_loader.dataset))
                seen_examples.append(examples_seen_so_far)

                # Switch back to training
                model.train()

                # Reset training loss
                train_loss = 0.

                # Update cached model and optimizer
                torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), RELATIVE_MODEL_LOC))
                torch.save(optimizer.state_dict(), os.path.join(os.path.dirname(__file__), RELATIVE_OPTIMIZER_LOC))

                print(f'Epoch {epoch} [{batch_index}]: {val_acc}')

    return model, train_loss_lst, val_accuracy_lst, seen_examples


def evaluate(model, data_loader) -> float:
    """Evaluate the Neural Network."""
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


def plot_results(seen_examples, statistics, plt_label, x_label, y_label) -> None:
    """Plot results."""
    # Plot results
    plt.plot(seen_examples, statistics, label=plt_label)

    # Set the x-axis and y-axis labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show(block=True)


def test_final_model(test_data_loader, relative_model_path, model_type) -> float:
    """
    Evaluate the final model on the test set.

    Preconditions:
        - model state is saved at relative_model_path
        - model_type is one of the string literals NUMERIC or LETTER
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
    train_data_loader, val_data_loader, test_data_loader = load_data(TRAIN_BATCH_SIZE, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE)

    # Train Code:

    # model, train_loss_lst, val_accuracy_lst, seen_examples = train_numeric(train_data_loader, val_data_loader, cache=True)
    # plot_results(seen_examples, train_loss_lst, 'Training Loss', 'Seen Examples', 'Loss')
    # plot_results(seen_examples, val_accuracy_lst, 'Validation Accuracy', 'Seen Examples', 'Accuracy')

    # Test Code:
    print(test_final_model(test_data_loader, RELATIVE_MODEL_LOC, 'NUMERIC'))






