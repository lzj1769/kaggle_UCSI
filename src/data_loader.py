import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from configure import SPLIT_FOLDER, DATA_FOLDER
from transform import *


def train_aug(image, mask):
    if np.random.rand() < 0.5:
        image, mask = do_horizontal_flip(image), do_horizontal_flip(mask)

    if np.random.rand() < 0.5:
        image, mask = do_vertical_flip(image), do_vertical_flip(mask)

    return image, mask


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
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(1400, 2100, order='F')

    return filename, masks


class CloudDataset(Dataset):
    def __init__(self, df, phase):
        self.df = df
        self.data_folder = DATA_FOLDER
        self.phase = phase
        self.filenames = self.df.ImageId.values

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.data_folder, image_id)
        image = cv2.imread(image_path) / 255.0
        image = do_resize_image(image=image, width=1024, height=1024)
        mask = do_resize_mask(mask=mask, width=1024, height=1024)

        if self.phase == "train":
            img, mask = train_aug(image=image, mask=mask)

        image, mask = img_to_tensor(image), mask_to_tensor(mask)
        mask = mask[0].permute(2, 0, 1)  # 1x4x1024x1024

        return image, mask

    def __len__(self):
        return len(self.filenames)


def get_dataloader(phase, fold, batch_size, num_workers):
    df_path = os.path.join(SPLIT_FOLDER, "fold_{}_{}.csv".format(fold, phase))
    df = pd.read_csv(df_path)
    image_dataset = CloudDataset(df, phase)
    shuffle = True if phase == "train" else False
    dataloader = DataLoader(image_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True,
                            shuffle=shuffle)

    return dataloader


if __name__ == '__main__':
    dataloader = get_dataloader(phase="train", fold=1, batch_size=10, num_workers=1)

    imgs, masks = next(iter(dataloader))

    print(imgs.shape)  # batch * 3 * 1024 * 1024
    print(masks)  # batch * 4 * 1024 * 1024
