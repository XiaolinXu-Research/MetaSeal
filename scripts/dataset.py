import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import sys
sys.path.append('./')
# import INN.config as c
from natsort import natsorted


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class Hinet_Dataset(Dataset):
    def __init__(self, config, transforms_=None, mode="train"):

        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            # train
            # self.files = natsorted(sorted(glob.glob(config.TRAIN_PATH + "/*." + config.format_train)))
            self.files_cover = natsorted(sorted(glob.glob(config.TRAIN_COVER_PATH + "/*." + config.format_train)))
            self.files_secret = natsorted(sorted(glob.glob(config.TRAIN_SECRET_PATH + "/*." + config.format_train)))
        else:
            # test
            # self.files = sorted(glob.glob(config.VAL_PATH + "/*." + config.format_val))
            self.files_cover = sorted(glob.glob(config.VAL_COVER_PATH + "/*.[jp][pn][g]"))
            self.files_secret = sorted(glob.glob(config.VAL_SECRET_PATH + "/*." + config.format_val))

    def __getitem__(self, index):
        try:
            image = Image.open(self.files_cover[index])
            secret = Image.open(self.files_secret[index])
            image = to_rgb(image)
            secret = to_rgb(secret)
            item = self.transform(image)
            secret = self.transform(secret)
            return item, secret

        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        if self.mode == 'shuffle':
            return min(len(self.files_cover), len(self.files_secret))

        else:
            return min(len(self.files_cover), len(self.files_secret))




def get_data_loaders(config, batch_size,  cropsize, mode):
    

    if mode == 'train':
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            # T.RandomCrop(cropsize),
            T.Resize((cropsize,cropsize)),
            T.ToTensor()
        ])
        dataloader = DataLoader(
            Hinet_Dataset(config, transforms_=transform, mode=mode),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            drop_last=True
    )
    else:

        transform = T.Compose([
            T.Resize((cropsize,cropsize)),
            T.ToTensor(),
        ])
    

        dataloader = DataLoader(
            Hinet_Dataset(config, transforms_=transform, mode=mode),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=1,
            drop_last=False
        )
    
    return dataloader 

