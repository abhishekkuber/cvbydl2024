from torch.utils.data import DataLoader
from dataloader import FilmPicturesDataset, lab_to_rgb
from rescale_dataset import loss as lossFunction
from architecture import FullNetwork
from utility import save_rgb_image, try_gpu, evaluate_accuracy
import torch
import cv2

TRAIN_DIR = 'small_set/train/Cinema'
TEST_DIR = 'small_set/test/Cinema'

def main():
    train_set = FilmPicturesDataset(TRAIN_DIR)
    test_set = FilmPicturesDataset(TEST_DIR)
    

    # print("test set at 0 has shapes \n INPUT", test_set[0][0].shape, "\n GROUND TRUTH:",test_set[0][1].shape)

    train_loader = DataLoader(dataset=train_set, batch_size=2, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)

    # Training parameters
    epochs = 10

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
        batch_loss = []
        for i, (x_batch, y_batch, luminance) in enumerate(train_loader):
            # Set to same device
            # x_batch -> torch.Size([batch_size, 1, 224, 224]) grayscale
            # y_batch -> torch.Size([batch_size, 2, 224, 224]) a,b
            # luminance -> torch.Size([batch_size, 224, 224])

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Set the gradients to zero TODO edit now since not sdg optimizer?
            optimizer.zero_grad()

            # Perform forward pass, y_pred now contains the predicted class and the predicted colors
            x_batch = x_batch.float()
            y_batch = y_batch.float()
                
            # y_pred shape = [batch_size, 2, 224, 224]
            y_pred = network(x_batch)[1]

            loss = criterion(y_pred, y_batch)
        
            '''luminance = train_set.get_luminance(i)
            y_pred = torch.squeeze(y_pred)
            y_pred_numpy = y_pred.cpu().detach().numpy()'''

            # Changing shape from [batch_size, 224, 224] to [batch_size, 1, 224, 224]
            luminance = luminance.unsqueeze(1)
            
            # needs you to detach, maybe move it after the backpropagation step?
            rgb = lab_to_rgb(luminance, y_pred)
            save_rgb_image(rgb, epoch, i)  # Implement save_rgb_image function
            
            # Compute the loss 
            print(f"Printing for each batch: Loss {loss}")

            # train_losses.append(loss)
            batch_loss.append(loss)

            # Backward computation and update
            loss.backward()
            optimizer.step()

        train_losses.append(sum(batch_loss)/ len(batch_loss))
        
        '''train_acc = evaluate_accuracy(train_loader, network.to('cpu'))
        test_acc = evaluate_accuracy(test_loader, network.to('cpu'))
        train_accs.append(train_acc)
        test_accs.append(test_acc)'''

        # Print performance
        # print('Epoch: {:.0f}'.format(epoch + 1))
        # print('Accuracy of train set: {:.00f}%'.format(train_acc))
        # print('Accuracy of test set: {:.00f}%'.format(test_acc))
        # print('')

    print(train_accs)
    print(test_accs)
    print(f"Training Losses : {train_losses}")

if __name__ == '__main__':
    main()