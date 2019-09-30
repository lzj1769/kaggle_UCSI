# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import os
import cv2
import numpy as np
import pandas as pd
from model import *

TEST_FOLDER = "../../input/understanding_cloud_organization/test_images"
TEST_DF = "../../input/understanding_cloud_organization/sample_submission.csv"

cls_name = ["Fish", "Flower", "Gravel", "Sugar"]


def do_horizontal_flip(image):
    # flip left-right
    image = cv2.flip(image, 1)
    return image


def img_to_tensor(img):
    tensor = torch.from_numpy(np.expand_dims(np.moveaxis(img, -1, 0), 0).astype(np.float32))
    return tensor


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


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
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


def optimal(valid_masks, probabilities):
    class_params = {}

    for class_id in range(4):
        print(class_id)
        attempts = []
        for t in range(0, 100, 5):
            t /= 100
            for ms in [0, 100, 1200, 5000, 10000]:
                masks = []
                for i in range(class_id, len(probabilities), 4):
                    probability = probabilities[i]
                    predict, num_predict = post_process(probability, t, ms)
                    masks.append(predict)

                d = []
                for i, j in zip(masks, valid_masks[class_id::4]):
                    if (i.sum() == 0) & (j.sum() == 0):
                        d.append(1)
                    else:
                        dice = (2.0 * (i * j).sum()) / (i.sum() + j.sum())
                        d.append(dice)

                attempts.append((t, ms, np.mean(d)))

        attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])

        attempts_df = attempts_df.sort_values('dice', ascending=False)
        print(attempts_df.head())
        best_threshold = attempts_df['threshold'].values[0]
        best_size = attempts_df['size'].values[0]

        class_params[class_id] = (best_threshold, best_size)

    return class_params


def segmentation_single_fold(fold):
    # prepare data
    df = pd.read_csv(TEST_DF)
    df['ImageId'] = df['Image_Label'].apply(lambda x: x.split('_')[0])
    filenames = df['ImageId'].unique().tolist()

    model = UResNet34(pretrained=False)
    model.cuda()

    model_save_path = "../../input/models/UResNet34/UResNet34_fold_{}.pt".format(fold)
    model.eval()
    state = torch.load(model_save_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])

    prediction = []
    for filename in filenames:
        image = cv2.imread(os.path.join(TEST_FOLDER, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 320), interpolation=cv2.INTER_LINEAR) / 255.0

        pred_seg = predict_segmentation(image=image, model=model)[0]
        for cls, pred in enumerate(pred_seg):
            pred = cv2.resize(pred, (525, 350), interpolation=cv2.INTER_LINEAR)
            pred, num = post_process(pred, threshold=0.5, min_size=10000)
            rle = mask2rle(pred.astype(np.int8))
            # rle = mask2rle((pred > 0.5).astype(np.int8))
            name = filename + "_" + cls_name[cls]
            prediction.append([name, rle])

    # save predictions to submission.csv
    submission_filename = "./submission/seg_fold_{}_valid_{}.csv".format(fold, state["best_dice"])
    message = "segmentation_fold_{}_valid_{}".format(fold, state["best_dice"])
    df = pd.DataFrame(prediction, columns=['Image_Label', 'EncodedPixels'])
    df.to_csv(submission_filename, index=False)
    submit(submission_filename=submission_filename, message=message)


if __name__ == '__main__':
    segmentation_single_fold(fold=2)
