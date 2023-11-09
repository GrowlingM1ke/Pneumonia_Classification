# import the necessary packages
from torch.utils.data import Dataset
import cv2
import numpy as np

class CNNDataset(Dataset):
    def __init__(self, normalPaths, pneumoniaPaths, transforms, subsample=False):
        # store the image and augmentation transforms
        self.normalPaths = normalPaths
        self.pneumoniaPaths = pneumoniaPaths
        self.transforms = transforms
        
        if not subsample:
            self.imagePaths = self.normalPaths + self.pneumoniaPaths
            self.labels = np.ones((len(self.imagePaths)))
            self.labels[0:len(self.normalPaths)] = 0
        
        else:
            # Subsample randomly the pneumonia paths
            self.random_subsample()

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)
    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]
        # load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask from disk in grayscale mode
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            mask = self.transforms(mask)
        # return a tuple of the image and its mask
        return (image, self.labels[idx])
    
    def random_subsample(self):
        self.imagePaths = self.normalPaths + np.random.choice(self.pneumoniaPaths, len(self.normalPaths), replace=False).tolist()