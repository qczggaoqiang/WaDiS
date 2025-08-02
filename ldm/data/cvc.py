import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import cv2
import random

def get_mean_and_std(img):
    x_mean, x_std = cv2.meanStdDev(img)  
    x_mean = np.hstack(np.around(x_mean, 2)) 
    x_std = np.hstack(np.around(x_std, 2))

    return x_mean, x_std

class CVCBase(Dataset):
    """CVC Dataset Base
    Notes:
        - `segmentation` is for the diffusion training stage (range binary -1 and 1)
        - `image` is for conditional signal to guided final seg-map (range -1 to 1)
    """
    def __init__(self, data_root, size=256, interpolation="nearest", mode=None, num_classes=2, color_aug=False):
        self.data_root = data_root
        self.mode = mode
        self.color_aug = color_aug
        assert mode in ["train", "val", "test"]
        self.data_paths = self._parse_data_list()
        self._length = len(self.data_paths)
        self.labels = dict(file_path_=[path for path in self.data_paths])
        self.size = size
        self.interpolation = dict(nearest=PIL.Image.NEAREST)[interpolation]   # for segmentation slice
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.CenterCrop(size=(256, 256))
        ])
        # TODO: more data transformation

        print(f"[Dataset]: CVC with 2 classes, in {self.mode} mode, use color_transfer:{self.color_aug}")

    def _color_aug(self, index, image):



        ##input_img
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)  # from RGB TO LAB
        image_mean, image_std = get_mean_and_std(image)  # get image_mean and image_std

        ##another_img
        all_index_list = list(range(len(self.data_paths)))
        all_index_list.pop(index)

        random.shuffle(all_index_list)
        choose_color_transfer_index = all_index_list[0]

        example = dict((k, self.labels[k][choose_color_transfer_index]) for k in self.labels)
        color_image = Image.fromarray(cv2.cvtColor(cv2.imread(example["file_path_"]), cv2.COLOR_BGR2RGB))
        color_image = color_image.resize((self.size, self.size), resample=PIL.Image.BILINEAR)
        color_image = cv2.cvtColor(np.array(color_image), cv2.COLOR_RGB2LAB)
        color_image_mean, color_image_std = get_mean_and_std(color_image)

        image_aug = (image - image_mean) / image_std * color_image_std + color_image_mean
        np.putmask(image_aug, image_aug > 255, 255)
        np.putmask(image_aug, image_aug < 0, 0)
        image_aug = cv2.cvtColor(cv2.convertScaleAbs(image_aug), cv2.COLOR_LAB2RGB)
        image_aug = Image.fromarray(image_aug)

        return image_aug

    """

    def _color_aug(self, index, image):

        ##input_img
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2YUV)  # from RGB TO YUV
        image_mean, image_std = get_mean_and_std(image)  # get image_mean and image_std

        ##another_img
        all_index_list = list(range(len(self.data_paths)))
        all_index_list.pop(index)

        random.shuffle(all_index_list)
        choose_color_transfer_index = all_index_list[0]

        example = dict((k, self.labels[k][choose_color_transfer_index]) for k in self.labels)
        color_image = Image.fromarray(cv2.cvtColor(cv2.imread(example["file_path_"]), cv2.COLOR_BGR2RGB))
        color_image = color_image.resize((self.size, self.size), resample=PIL.Image.BILINEAR)
        color_image = cv2.cvtColor(np.array(color_image), cv2.COLOR_RGB2YUV)
        color_image_mean, color_image_std = get_mean_and_std(color_image)

        image_aug = (image - image_mean) / image_std * color_image_std + color_image_mean
        np.putmask(image_aug, image_aug > 255, 255)
        np.putmask(image_aug, image_aug < 0, 0)
        image_aug = cv2.cvtColor(cv2.convertScaleAbs(image_aug), cv2.COLOR_YUV2RGB)
        image_aug = Image.fromarray(image_aug)

        return image_aug
   """
    def __getitem__(self, i):
        # read segmentation and images
        example = dict((k, self.labels[k][i]) for k in self.labels)
        # segmentation = Image.open(example["file_path_"].replace("Original", "GroundTruth")).convert("RGB")
        # image = Image.open(example["file_path_"]).convert("RGB")    # same name, different postfix
        segmentation = Image.fromarray(cv2.cvtColor(cv2.imread(example["file_path_"].replace("Original", "Ground Truth")),cv2.COLOR_BGR2RGB))
        image = Image.fromarray(cv2.cvtColor(cv2.imread(example["file_path_"]),cv2.COLOR_BGR2RGB))

        if self.size is not None:
            segmentation = segmentation.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            image = image.resize((self.size, self.size), resample=PIL.Image.BILINEAR)

        
        # image.save('ori.jpg')
        
        if self.mode == 'train' and self.color_aug:
            image = self._color_aug(i, image)
            # image.save('aug.jpg')
        
        if self.mode == "train":
            segmentation, image = self._utilize_transformation(segmentation, image, self.transform)

        segmentation = (np.array(segmentation) > 128).astype(np.float32)
        if self.mode == "test":
            example["segmentation"] = segmentation   
        else:
            example["segmentation"] = ((segmentation * 2) - 1)   # range: binary -1 and 1

        image = np.array(image).astype(np.float32) / 255.
        image = (image * 2.) - 1.                            # range from -1 to 1, np.float32
        example["image"] = image
        example["class_id"] = np.array([-1])  # doesn't matter for binary seg

        assert np.max(segmentation) <= 1. and np.min(segmentation) >= -1.
        assert np.max(image) <= 1. and np.min(image) >= -1.
        return example

    def __len__(self):
        return self._length

    def _parse_data_list(self): # 80% / 10% / 10%
        all_imgs = glob.glob(os.path.join(self.data_root, "*.png"))
        train_imgs, val_imgs, test_imgs = all_imgs[:492], all_imgs[492:492+60], all_imgs[492+60:]

        if self.mode == "train":
            return train_imgs
        elif self.mode == "val":
            return val_imgs
        elif self.mode == "test":
            return test_imgs
        else:
            raise NotImplementedError(f"Only support dataset split: train, val, test !")

    @staticmethod
    def _utilize_transformation(segmentation, image, func):
        state = torch.get_rng_state()
        segmentation = func(segmentation)
        torch.set_rng_state(state)
        image = func(image)
        return segmentation, image


class CVCTrain(CVCBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/CVC/PNG/Original", mode="train", **kwargs)


class CVCValidation(CVCBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/CVC/PNG/Original", mode="val", **kwargs)


class CVCTest(CVCBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/CVC/PNG/Original", mode="test", **kwargs)

if __name__ == '__main__':
    cvc_train = CVCTrain(color_aug=True)
    
    sample = cvc_train[0]