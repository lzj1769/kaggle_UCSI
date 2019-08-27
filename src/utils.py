import torch


def compute_dice(preds, truth, threshold=0.5):
    probability = torch.sigmoid(preds)
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    mean_dice_channel = 0.
    with torch.no_grad():
        for i in range(batch_size):
            for j in range(channel_num):
                channel_dice = dice_single_channel(probability[i, j, :, :], truth[i, j, :, :], threshold)

                mean_dice_channel += channel_dice / (batch_size * channel_num)

    return mean_dice_channel


def dice_single_channel(probability, truth, threshold, eps=1E-9):
    p = (probability.view(-1) > threshold).float()
    t = (truth.view(-1) > 0.5).float()
    dice = (2.0 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
    return dice


def dice_loss(preds, truth):
    probability = torch.sigmoid(preds)
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    loss = 0.0
    for i in range(batch_size):
        for j in range(channel_num):
            loss += dice_loss_single_channel(probability[i, j, :, :], truth[i, j, :, :]) / (batch_size * channel_num)

    return loss


def dice_loss_single_channel(probability, true, smooth=1.0):
    iflat = probability.view(-1)
    tflat = true.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


if __name__ == '__main__':
    from data_loader import get_dataloader
    from torch.nn import BCEWithLogitsLoss
    from model import UResNet34

    dataloader = get_dataloader(phase="train", fold=0, batch_size=4, num_workers=2)
    model = UResNet34()
    model.cuda()
    model.train()
    imgs, masks = next(iter(dataloader))
    imgs, masks = next(iter(dataloader))
    preds = model(imgs.cuda())
    criterion = BCEWithLogitsLoss()
    loss = dice_loss(preds, masks.cuda())
    print(loss.item())
    loss = criterion(preds, masks.cuda())
    print(loss.item())
