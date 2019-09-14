import os
import time
import argparse
import numpy as np
import pandas as pd
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from model import *
from data_loader import get_dataloader
from configure import SAVE_MODEL_PATH, TRAINING_HISTORY_PATH, SPLIT_FOLDER
from loss import DiceBCELoss
from utils import seed_torch, compute_dice
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Training model for steel defect detection')
    parser.add_argument("--model", type=str, default='UResNet34',
                        help="Name for encode used in Unet. Currently available: UResNet34")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of workers for training. Default: 2")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training. Default: 4")
    parser.add_argument("--num-epochs", type=int, default=100,
                        help="Number of epochs for training. Default: 100")
    parser.add_argument("--fold", type=int, default=0)

    return parser.parse_args()


class Trainer(object):
    def __init__(self, model, num_workers, batch_size, num_epochs, model_save_path, model_save_name,
                 fold, training_history_path):
        self.model = model
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.best_loss = np.inf
        self.phases = ["train", "valid"]
        self.model_save_path = model_save_path
        self.model_save_name = model_save_name
        self.fold = fold
        self.training_history_path = training_history_path
        self.criterion = DiceBCELoss()

        self.optimizer = SGD(self.model.parameters(), lr=1e-02, momentum=0.9, weight_decay=1e-04)
        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.model = self.model.cuda()
        self.dataloaders = {
            phase: get_dataloader(
                phase=phase,
                fold=fold,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.loss = {phase: [] for phase in self.phases}
        self.bce_loss = {phase: [] for phase in self.phases}
        self.dice_loss = {phase: [] for phase in self.phases}
        self.dice = {phase: [] for phase in self.phases}

    def forward(self, images, masks):
        outputs = self.model(images.cuda())
        loss, bce_loss, dice_loss = self.criterion(outputs, masks.cuda())
        return loss, bce_loss, dice_loss, outputs

    def iterate(self, phase):
        self.model.train(phase == "train")

        running_loss = 0.0
        running_bce_loss = 0.0
        running_dice_loss = 0.0

        for images, masks in self.dataloaders[phase]:
            loss, bce_loss, dice_loss, outputs = self.forward(images, masks)
            if phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            running_bce_loss += bce_loss.item()
            running_dice_loss += dice_loss.item()

        epoch_loss = running_loss / len(self.dataloaders[phase])
        epoch_bce_loss = running_bce_loss / len(self.dataloaders[phase])
        epoch_dice_loss = running_dice_loss / len(self.dataloaders[phase])

        self.loss[phase].append(epoch_loss)
        self.bce_loss[phase].append(epoch_bce_loss)
        self.dice_loss[phase].append(epoch_dice_loss)

        torch.cuda.empty_cache()

        return epoch_loss, epoch_bce_loss, epoch_dice_loss

    def plot_history(self):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        ax1.plot(self.loss['train'], '-b', label='Training')
        ax1.plot(self.loss['valid'], '-r', label='Validation')
        ax1.set_title("Loss", fontweight='bold')
        ax1.legend(loc="upper right", frameon=False)

        ax2.plot(self.bce_loss['train'], '-b', label='Training')
        ax2.plot(self.bce_loss['valid'], '-r', label='Validation')
        ax2.set_title("BCE Loss", fontweight='bold')
        ax2.legend(loc="upper right", frameon=False)

        ax3.plot(self.dice_loss['train'], '-b', label='Training')
        ax3.plot(self.dice_loss['valid'], '-r', label='Validation')
        ax3.set_title("Dice Loss", fontweight='bold')
        ax3.legend(loc="upper right", frameon=False)

        output_filename = os.path.join(self.training_history_path,
                                       "{}_fold_{}_loss.pdf".format(self.model_save_name, self.fold))
        fig.tight_layout()
        fig.savefig(output_filename)

        output_filename = os.path.join(self.training_history_path,
                                       "{}_fold_{}_loss.txt".format(self.model_save_name, self.fold))
        header = ["Training loss", "Validation loss",
                  "Training bce loss", "Validation loss",
                  "Training dice loss", "Validation dice loss"]

        with open(output_filename, "w") as f:
            f.write("\t".join(header) + "\n")
            for i in range(len(self.loss['train'])):
                res = [self.loss['train'][i], self.loss['valid'][i],
                       self.bce_loss['train'][i], self.bce_loss['valid'][i],
                       self.dice_loss['train'][i], self.dice_loss['valid'][i]]

                f.write("\t".join(map(str, res)) + "\n")

    def plot_dice(self, thresholds, mean_dice):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

        axes[0, 0].plot(thresholds, mean_dice[:, 0], '-b')
        axes[0, 0].set_title("Class 1", fontweight='bold')

        axes[0, 1].plot(thresholds, mean_dice[:, 1], '-b')
        axes[0, 1].set_title("Class 2", fontweight='bold')

        axes[1, 0].plot(thresholds, mean_dice[:, 2], '-b')
        axes[1, 0].set_title("Class 3", fontweight='bold')

        axes[1, 1].plot(thresholds, mean_dice[:, 3], '-b')
        axes[1, 1].set_title("Class 4", fontweight='bold')

        output_filename = os.path.join(self.training_history_path,
                                       "{}_fold_{}_dice.pdf".format(self.model_save_name, self.fold))
        fig.tight_layout()
        fig.savefig(output_filename)

        output_filename = os.path.join(self.training_history_path,
                                       "{}_fold_{}_dice.txt".format(self.model_save_name, self.fold))

        header = ["Threshold", "Class 1", "Class 2", "Class 3", "Class 4"]

        with open(output_filename, "w") as f:
            f.write("\t".join(header) + "\n")
            for i in range(len(thresholds)):
                res = [thresholds[i], mean_dice[i, 0], mean_dice[i, 1], mean_dice[i, 2], mean_dice[i, 3]]
                f.write("\t".join(map(str, res)) + "\n")

    def start(self):
        for epoch in range(self.num_epochs):
            start = time.strftime("%D-%H:%M:%S")
            print("Epoch: {}/{} |  time : {}".format(epoch + 1, self.num_epochs, start))
            print("Learning rate: %0.8f" % self.scheduler.get_lr()[0])
            print("=================================================================")

            train_loss, train_bce_loss, train_dice_loss = self.iterate("train")
            with torch.no_grad():
                valid_loss, valid_bce_loss, valid_dice_loss = self.iterate("valid")

            print("train_loss: %0.8f, train_bce_loss: %0.8f, train_dice_loss: %0.8f" % (train_loss, train_bce_loss,
                                                                                        train_dice_loss))
            print("valid_loss: %0.8f, valid_bce_loss: %0.8f, valid_dice_loss: %0.8f" % (valid_loss, valid_bce_loss,
                                                                                        valid_dice_loss))

            self.scheduler.step(epoch=epoch)
            if valid_loss < self.best_loss:
                print("******** Validation loss improved from %0.8f to %0.8f ********" % (self.best_loss, valid_loss))
                self.best_loss = valid_loss
                if epoch > 10:
                    thresholds, best_dice = self.optimize_threshold()
                    print("******** Optimized thresholds: %0.8f, %0.8f, %0.8f, %0.8f ********" % (thresholds[0],
                                                                                                  thresholds[1],
                                                                                                  thresholds[2],
                                                                                                  thresholds[3]))
                    print("******** Best dices:           %0.8f, %0.8f, %0.8f, %0.8f ********" % (best_dice[0],
                                                                                                  best_dice[1],
                                                                                                  best_dice[2],
                                                                                                  best_dice[3]))
                    print("******** Mean dice:            %0.8f" % np.mean(best_dice))
                    state = {
                        "threshold": thresholds,
                        "best_dice": best_dice,
                        "state_dict": self.model.state_dict(),
                    }
                else:
                    state = {
                        "state_dict": self.model.state_dict(),
                    }

                filename = os.path.join(self.model_save_path, "{}_fold_{}.pt".format(self.model_save_name, self.fold))
                if os.path.exists(filename):
                    os.remove(filename)
                torch.save(state, filename)

            print()
            self.plot_history()

    def optimize_threshold(self):
        mean_dice = np.zeros(shape=(100, 4))
        thresholds = np.linspace(start=0, stop=1, num=100)
        for images, masks in self.dataloaders["valid"]:
            preds = self.model(images.cuda()).detach().cpu()
            for i, threshold in enumerate(thresholds):
                dice = compute_dice(preds, masks, threshold=threshold)
                for j in range(4):
                    mean_dice[i, j] += dice[j].item()

        mean_dice = mean_dice / len(self.dataloaders["valid"])
        best_dice = np.max(mean_dice, axis=0)
        best_dice_index = np.argmax(mean_dice, axis=0)

        self.plot_dice(thresholds, mean_dice)

        return thresholds[best_dice_index], best_dice


def main():
    args = parse_args()

    seed_torch(seed=42)

    model = None
    if args.model == "UResNet34":
        model = UResNet34()

    model_save_path = os.path.join(SAVE_MODEL_PATH, args.model)
    training_history_path = os.path.join(TRAINING_HISTORY_PATH, args.model)

    df_train_path = os.path.join(SPLIT_FOLDER, "fold_{}_train.csv".format(args.fold))
    df_train = pd.read_csv(df_train_path)

    df_valid_path = os.path.join(SPLIT_FOLDER, "fold_{}_valid.csv".format(args.fold))
    df_valid = pd.read_csv(df_valid_path)

    print("Training on {} images, Fish: {}, Flower: {}, Gravel: {}, Sugar: {}".format(len(df_train),
                                                                                      df_train['isFish'].sum(),
                                                                                      df_train['isFlower'].sum(),
                                                                                      df_train['isGravel'].sum(),
                                                                                      df_train['isSugar'].sum()))
    print("Validate on {} images, Fish: {}, Flower: {}, Gravel: {}, Sugar: {}".format(len(df_valid),
                                                                                      df_valid['isFish'].sum(),
                                                                                      df_valid['isFlower'].sum(),
                                                                                      df_valid['isGravel'].sum(),
                                                                                      df_valid['isSugar'].sum()))

    model_trainer = Trainer(model=model,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            num_epochs=args.num_epochs,
                            model_save_path=model_save_path,
                            training_history_path=training_history_path,
                            model_save_name=args.model,
                            fold=args.fold)
    model_trainer.start()


if __name__ == '__main__':
    main()
