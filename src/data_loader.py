import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from configure import SPLIT_FOLDER, DATA_FOLDER
from albumentations import (RandomRotate90, Transpose, ShiftScaleRotate, Flip, Compose)


def img_to_tensor(img):
    tensor = torch.from_numpy(np.moveaxis(img, -1, 0).astype(np.float32)) / 255.0
    return tensor


def mask_to_tensor(mask):
    return torch.from_numpy(mask.astype(np.float32))


def train_aug(p=0.5):
    return Compose([RandomRotate90(), Flip(), Transpose(),
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45)], p=p)


def make_mask(row_id, df):
    # Given a row index, return image_id and mask (1400, 2100, 4)
    filename = df.iloc[row_id].ImageId
    labels = df.iloc[row_id][1:5]

    masks = np.zeros((1400, 2100, 4), dtype=np.float32)  # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)
    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(1400 * 2100, dtype=np.uint8)
            for pos, le in zip(positions, length):
                pos -= 1
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(1400, 2100, order='F')

    return filename, masks


class CloudDataset(Dataset):
    def __init__(self, df, phase):
        self.df = df
        self.data_folder = DATA_FOLDER
        self.phase = phase
        self.filenames = self.df.ImageId.values
        self.transform = train_aug(p=0.5)

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.data_folder, image_id)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (1024, 1024))
        mask = cv2.resize(mask, (1024, 1024))
        mask = (mask > 0.5).astype(np.int8)

        if self.phase == "train":
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image, mask = img_to_tensor(image), mask_to_tensor(mask)
        mask = mask.permute(2, 0, 1)  # 4x1024x1024

        return image, mask

    def __len__(self):
        return len(self.filenames)


def get_dataloader(phase, fold, batch_size, num_workers):
    df_path = os.path.join(SPLIT_FOLDER, "fold_{}_{}.csv".format(fold, phase))
    df = pd.read_csv(df_path)
    image_dataset = CloudDataset(df, phase)
    shuffle = True if phase == "train" else False
    drop_last = True if phase == "train" else False
    dataloader = DataLoader(image_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True,
                            shuffle=shuffle,
                            drop_last=drop_last)

    return dataloader


if __name__ == '__main__':
    dataloader = get_dataloader(phase="train", fold=1, batch_size=10, num_workers=1)

    imgs, masks = next(iter(dataloader))

    print(imgs.shape)  # batch * 3 * 1024 * 1024
    print(masks)  # batch * 4 * 1024 * 1024
