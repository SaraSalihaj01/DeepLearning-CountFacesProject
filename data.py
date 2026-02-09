import torch, random    #Pytorch to manage tensors,dataset and training
import pandas as pd     #pandas -> serves to read and manage CSV files containing the dataset (image paths and face count)
from PIL import Image   #is a library for opening and manipulating images
from torchvision import transforms    #transforms modules to perform transformations on images (resize,flip,jitter,etc.)

"""This CountFacesDataset class reads a CSV with image paths and face counts, opens each image, transforms it into a tensor(with 
possible data augmentation) and returs (image,count) for model training."""
class CountFacesDataset(torch.utils.data.Dataset):   #the class extend torch.utils.data.Dataset and implement __len__ and __getitem__
    """Dataset for face counting.

    Reads a CSV file with two columns:
        - 'path': image's path
        - 'count': number of faces in the image (int)

    Each item returns (x, y):
        - x: image tensor (float32) after preprocessing/augmentation
        - y: target tensor (float32) with the face count (shape: [1])

    If 'augment=True', additional transformations (crop/flip/jitter) are applied
    to increase robustness during training.
    """
    def __init__(self, csv_path, img_size=256, augment=False):  #initialize the dataset
        self.df = pd.read_csv(csv_path)  #csv_path(str)-> path to CSV file containing 'path' and 'columns'
        self.img_size = img_size         #image_size(int)-> size in pixels to which each image is resized, defaults to 256
        self.augment = augment           #augment(bool)->if TRUE ->apply data augmentation, defaults to FALSE  
        t_train = [                                   
            transforms.Resize((img_size, img_size)),   #Resize the image to (img_size,img_size) pixels 
            transforms.ToTensor(),                     #Convert the image to a PyTorch tensor and scale pixel values to [0,1]
            transforms.ConvertImageDtype(torch.float32), #Ensure the tensor has dtype float32(standard for training)
        ]
        t_aug = [
            transforms.Resize(int(img_size*1.1)),  #Enlarge the image slightly at 10% bigger
            transforms.RandomResizedCrop(img_size, scale=(0.8,1.0), ratio=(0.9,1.1)),  #Randomly crop and resize back to (img_size,img_size)
            transforms.RandomHorizontalFlip(),  #Flip the image horizontally (50% of probability)
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02), #Randomly change brightness,contrast, saturation and hue to help the model generalize to different lighting/color conditions.
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ]
        self.t = transforms.Compose(t_aug if augment else t_train)  #Choose the pipeline: if augment=TRUE-> apply data augmentation, otherwise apply only simple preprocessing (t_train).                         
        #In this way we can use the same class for train and validation, changing only t5he flag augmentation.
                                                                        
    def __len__(self):     #Return the number of samples in the dataset (total number of rows in the CSV file)
        return len(self.df)

    def __getitem__(self, idx):   #load and transform a single sample with index(idx)
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        x = self.t(img)                #Image tensor of shape(C,H,W) dtype float32
        y = torch.tensor([float(row['count'])], dtype=torch.float32)  #Target tensor of shape[1], dtype float32(face count)
        return x, y                     #Returns: tuple[torch.Tensor, torch.Tensor]
