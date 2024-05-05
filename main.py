from torch.utils.data import DataLoader
from dataloader import FilmPicturesDataset
from rescale_dataset import loss as lossFunction
from architecture import FullNetwork
from utility import try_gpu, evaluate_accuracy
import torch

TRAIN_DIR = 'small_set/train/Cinema'
TEST_DIR = 'small_set/test/Cinema'

def main():
    train_set = FilmPicturesDataset(TRAIN_DIR)
    test_set = FilmPicturesDataset(TEST_DIR)

    print("test set at 0 has shapes \n INPUT", test_set[0][0].shape, "\n GROUND TRUTH:",test_set[0][1].shape)

    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)

    # Training parameters
    epochs = 3

    # Initialize network
    network = FullNetwork()
    optimizer = torch.optim.Adadelta(network.parameters())
    criterion = lossFunction

    # Define list to store losses and performances of each iteration
    train_losses = []
    train_accs = []
    test_accs = []

    # Try using gpu instead of cpu
    device = try_gpu()

    for epoch in range(epochs):

        # Network in training mode and to device
        network.train()
        network.to(device)

        # Training loop
        for i, (x_batch, y_batch) in enumerate(train_loader):
            # Set to same device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Set the gradients to zero TODO edit now since not sdg optimizer?
            optimizer.zero_grad()

            # Perform forward pass, y_pred now contains the predicted class and the predicted colors
            x_batch = x_batch.to(torch.float32)
            y_pred = network(x_batch)

            # Compute the loss
            loss = criterion(y_pred[1], y_batch)
            train_losses.append(loss)

            # Backward computation and update
            loss.backward()
            optimizer.step()

        # Compute train and test error
        train_acc = 100 * evaluate_accuracy(train_loader, network.to('cpu'))
        test_acc = 100 * evaluate_accuracy(test_loader, network.to('cpu'))

        # Development of performance
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # Print performance
        print('Epoch: {:.0f}'.format(epoch + 1))
        print('Accuracy of train set: {:.00f}%'.format(train_acc))
        print('Accuracy of test set: {:.00f}%'.format(test_acc))
        print('')


if __name__ == '__main__':
    main()