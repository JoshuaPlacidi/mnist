import matplotlib.pyplot as plt
import torch

def show_image(image):

    if type(image) == torch.Tensor:
        image = image.numpy()

    plt.imshow(image.squeeze())
    plt.show()
