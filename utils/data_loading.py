import numpy as np
# import torch
from torch.utils.data import Dataset
# import torchvision.transforms as transforms


# transform = transforms.Compose([
#     transforms.ToTensor()
# ])


class CustomDataset(Dataset):

    def __init__(self, images, masks, transforms=None): 
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __getitem__(self, index):
        image = self.images[index]
        # image = np.asarray(image).astype(np.float64).reshape(28, 28, 1)
        mask = self.masks[index]
        # mask = np.asarray(mask).astype(np.float64).reshape(28, 28, 1)
        if self.transforms:
            image = self.transforms(image)
      
        return image, mask

    def __len__(self):  

       return len(self.images)