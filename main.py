from torch.utils.data import DataLoader
from dataloader import FilmPicturesDataset, lab_to_rgb
from rescale_dataset import loss as lossFunction
from architecture import FullNetwork
from copied_architecture import ColNet
from utility import save_rgb_image, try_gpu, evaluate_accuracy
import torch
import cv2

TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'

def main():
    train_set = FilmPicturesDataset(TRAIN_DIR)
    test_set = FilmPicturesDataset(TEST_DIR)
    
    train_loader = DataLoader(dataset=train_set, batch_size=10, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=10, shuffle=True)

    # Training parameters
    epochs = 25
    
    # TODO: same images, but try to multiply it 100 times
    
    # Initialize network
    network = FullNetwork()
    # network = ColNet(num_classes=3)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    # criterion = lossFunction
    criterion = torch.nn.MSELoss(reduction='sum')

    # Define list to store losses and performances of each iteration
    train_losses = []
    train_accs = []
    test_accs = []

    # Try using gpu instead of cpu
    device = try_gpu()
    
    network.train()
    network.to(device)
    
    for epoch in range(epochs):

        # Network in training mode and to device
        

        # Training loop
        epoch_loss = 0.0
        for i, (x_batch, y_batch) in enumerate(train_loader):
            
            # x_batch = torch.Size([2, 1, 224, 224]) Luminance channels
            # y_batch = torch.Size([2, 2, 224, 224]) a,b channels
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)


            optimizer.zero_grad()

            # Only take index 1 because that is the output of the colorization network
            y_pred = network(x_batch)
            
            loss = criterion(y_pred, y_batch)
            
            loss.backward()
            optimizer.step()
            
            # Compute the loss 
            print(f"Epoch : {epoch} , Batch : {i}, Loss : {loss.item()}")

            # train_losses.append(loss)
            epoch_loss += loss.item()

            # Backward computation and update
            
        # Divide the total loss by the number of batches
        epoch_loss /= len(train_loader) 
        train_losses.append(epoch_loss)

    
    # Training done, test the model now
    network.eval()
    network.to(device)
    test_loss = 0.0
    with torch.no_grad():
        print(f"Testing the model now")
        for i, (x_batch, y_batch) in enumerate(test_loader):
            
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            y_pred = network(x_batch)
            loss = criterion(y_pred, y_batch)
            test_loss += loss.item()
            
            y_pred = y_pred.to(torch.device("cpu"))

            for k in range(y_pred.shape[0]):
                rgb = lab_to_rgb(x_batch[k], y_pred[k])
                
                save_rgb_image(rgb, i, k)
            
            # Compute the loss 
            
            


    print(train_accs)
    print(test_accs)
    print(f"Training Losses : {train_losses}")

if __name__ == '__main__':
    main()