import os
import cv2
import numpy as np
import pandas as pd
import argparse
from model import *

TEST_FOLDER = "../../input/understanding_cloud_organization/test_images"
TEST_DF = "../../input/understanding_cloud_organization/sample_submission.csv"
TRAIN_DATA_FOLDER = "../../input/understanding_cloud_organization/train_images"
SPLIT_FOLDER = "../../input/understanding_cloud_organization/split"

cls_name = ["Fish", "Flower", "Gravel", "Sugar"]


def parse_args():
    parser = argparse.ArgumentParser(description='Training model for steel defect detection')
    parser.add_argument("--fold", type=int, default=0)

    return parser.parse_args()


def do_horizontal_flip(image):
    # flip left-right
    image = cv2.flip(image, 1)
    return image


def img_to_tensor(img):
    tensor = torch.from_numpy(np.expand_dims(np.moveaxis(img, -1, 0), 0).astype(np.float32))
    return tensor


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


# https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def post_process(mask, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    num_component, component = cv2.connectedComponents(mask)
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def rle2mask(rle=""):
    if rle == "":
        return np.zeros((320, 640))
    else:
        label = rle.split(" ")
        positions = map(int, label[0::2])
        length = map(int, label[1::2])
        mask = np.zeros(320 * 640, dtype=np.uint8)
        for pos, le in zip(positions, length):
            pos -= 1
            mask[pos:(pos + le)] = 1

        mask = mask.reshape(320, 640, order='F')

        return mask


def submit(submission_filename, message):
    command = "kaggle competitions submit -c understanding_cloud_organization -f"
    os.system(command + " " + submission_filename + " -m " + message)


def predict_segmentation(image, model):
    image_raw = img_to_tensor(image)
    preds_raw = torch.sigmoid(model(image_raw.cuda()))
    preds_raw = preds_raw.detach().cpu().numpy()

    image_hflip = img_to_tensor(do_horizontal_flip(image))
    preds_hflip = torch.sigmoid(model(image_hflip.cuda()))
    preds_hflip = preds_hflip.detach().cpu().numpy()

    # we need to flip the prediction back
    for cls in range(4):
        preds_hflip[0, cls, :, :] = do_horizontal_flip(preds_hflip[0, cls, :, :])

    preds = (preds_raw + preds_hflip) / 2

    return preds


def predict_classification(image, model):
    image_raw = img_to_tensor(image)
    preds_raw = torch.sigmoid(model(image_raw.cuda()))
    preds_raw = preds_raw.detach().cpu().numpy()

    image_hflip = img_to_tensor(do_horizontal_flip(image))
    preds_hflip = torch.sigmoid(model(image_hflip.cuda()))
    preds_hflip = preds_hflip.detach().cpu().numpy()

    preds = (preds_raw + preds_hflip) / 2

    return preds


def optimal(valid_masks, valid_preds):
    num_images, classes, height, width = valid_masks.shape

    class_params, valid_dice = {}, 0
    for cls in range(classes):
        attempts = []
        for threshold in range(0, 100, 5):
            threshold /= 100
            for min_size in [0, 100, 1200, 5000, 10000]:
                dice = 0
                for i in range(num_images):
                    pred_mask = (valid_preds[i, cls] > threshold).astype(np.int8)
                    pred_mask, num_predict = post_process(pred_mask, min_size)

                    p = pred_mask
                    t = valid_masks[i, cls]

                    if p.sum() == 0.0 and t.sum() == 0.0:
                        dice += 1.0
                    else:
                        dice += (2.0 * (p * t).sum()) / (p.sum() + t.sum())

                dice /= num_images
                attempts.append((threshold, min_size, dice))

        attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])

        attempts_df = attempts_df.sort_values('dice', ascending=False)
        print(attempts_df.head())
        best_threshold = attempts_df['threshold'].values[0]
        best_size = attempts_df['size'].values[0]
        valid_dice += attempts_df['dice'].values[0]

        class_params[cls] = (best_threshold, best_size)

    return class_params, valid_dice / 4


def segmentation_single_fold_valid(model, fold):
    # prepare data
    df_path = os.path.join(SPLIT_FOLDER, "fold_{}_valid.csv".format(fold))
    df = pd.read_csv(df_path)

    valid_masks = np.zeros(shape=(len(df), 4, 350, 525), dtype=np.float32)
    valid_preds = np.zeros(shape=(len(df), 4, 350, 525), dtype=np.float32)

    for i in range(len(df)):
        filename, mask = make_mask(i, df)
        image = cv2.imread(os.path.join(TRAIN_DATA_FOLDER, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 320), interpolation=cv2.INTER_LINEAR) / 255.0

        pred_seg = predict_segmentation(image=image, model=model)[0]
        for cls, pred in enumerate(pred_seg):
            valid_masks[i, cls] = cv2.resize(mask[:, :, cls], (525, 350), interpolation=cv2.INTER_LINEAR)
            valid_masks[i, cls] = (valid_masks[i, cls] > 0.5).astype(np.float32)
            valid_preds[i, cls] = cv2.resize(pred, (525, 350), interpolation=cv2.INTER_LINEAR)

    return valid_masks, valid_preds


def segmentation_single_fold_test(model, fold, class_params=None, valid_dice=None):
    # prepare data
    df = pd.read_csv(TEST_DF)
    df['ImageId'] = df['Image_Label'].apply(lambda x: x.split('_')[0])
    filenames = df['ImageId'].unique().tolist()

    prediction = []
    for filename in filenames:
        image = cv2.imread(os.path.join(TEST_FOLDER, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 320), interpolation=cv2.INTER_LINEAR) / 255.0

        pred_seg = predict_segmentation(image=image, model=model)[0]
        for cls, pred in enumerate(pred_seg):
            pred = cv2.resize(pred, (525, 350), interpolation=cv2.INTER_LINEAR)
            if class_params is not None:
                pred = (pred > class_params[cls][0]).astype(np.int8)
                pred, num = post_process(pred, min_size=class_params[cls][1])
                rle = mask2rle(pred.astype(np.int8))
            else:
                rle = mask2rle((pred > 0.5).astype(np.int8))

            name = filename + "_" + cls_name[cls]
            prediction.append([name, rle])

    # save predictions to submission.csv
    submission_filename = "./submission/seg_fold_{}_valid_{}.csv".format(fold, valid_dice)
    message = "segmentation_fold_{}_valid_{}".format(fold, valid_dice)
    df = pd.DataFrame(prediction, columns=['Image_Label', 'EncodedPixels'])
    df.to_csv(submission_filename, index=False)
    # submit(submission_filename=submission_filename, message=message)


def segmentation_single_fold(fold):
    model = UResNet34(pretrained=False)
    model.cuda()

    model_save_path = "../../input/models/UResNet34/UResNet34_fold_{}.pt".format(fold)
    model.eval()
    state = torch.load(model_save_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])

    valid_masks, valid_preds = segmentation_single_fold_valid(model=model, fold=fold)

    # optimize threshold and mask size
    class_params, valid_dice = optimal(valid_masks=valid_masks, valid_preds=valid_preds)
    segmentation_single_fold_test(model=model, fold=fold, class_params=class_params, valid_dice=valid_dice)


def main():
    args = parse_args()
    segmentation_single_fold(fold=args.fold)


if __name__ == '__main__':
    main()
