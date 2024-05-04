from torch.utils.data import DataLoader
from dataloader import FilmPicturesDataset

TRAIN_DIR = 'rescaled/train/Cinema'
TEST_DIR = 'rescaled/test/Cinema'


def main():
    train_set = FilmPicturesDataset(TRAIN_DIR)
    test_set = FilmPicturesDataset(TEST_DIR)

    print("test set at 0 has shapes \n INPUT", test_set[0][0].shape, "\n GROUND TRUTH:",test_set[0][1].shape)

    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)


if __name__ == '__main__':
    main()