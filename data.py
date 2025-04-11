import torch
import os
from utils import show_image
import idx2numpy
import numpy as np

class MNISTDataset(torch.utils.data.Dataset):

    def __init__(self, images_path: str, labels_path: str):

        self.images = np.array(idx2numpy.convert_from_file(images_path), copy=True)
        self.labels = np.array(idx2numpy.convert_from_file(labels_path), copy=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert to tensors - no need to copy since we already have writable arrays
        image = torch.from_numpy(self.images[idx]).float()
        # Convert label to numpy array explicitly to handle scalar values
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        return image, label

def get_dataloaders(
    train_images_path: str,
    train_labels_path: str,
    test_images_path: str,
    test_labels_path: str,
    batch_size: int,
    num_workers: int,
):

    train_dataset = MNISTDataset(train_images_path, train_labels_path)
    test_dataset = MNISTDataset(test_images_path, test_labels_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, test_loader

if __name__ == "__main__":

    import idx2numpy

    # Load training data
    # train_images = idx2numpy.convert_from_file('train-images-idx3-ubyte')
    # train_labels = idx2numpy.convert_from_file('train-labels-idx1-ubyte')

    # Load test data
    test_images = idx2numpy.convert_from_file(os.path.expanduser('~/data/mnist/t10k-images.idx3-ubyte'))
    test_labels = idx2numpy.convert_from_file(os.path.expanduser('~/data/mnist/t10k-labels.idx1-ubyte'))

    print(test_images.shape)
    print(test_labels.shape)

    show_image(test_images[0])
