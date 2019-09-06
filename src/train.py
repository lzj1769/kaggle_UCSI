import os
import time
import argparse
import numpy as np
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import *
from data_loader import get_dataloader
from configure import SAVE_MODEL_PATH, TRAINING_HISTORY_PATH
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
    parser.add_argument("--num-epochs", type=int, default=200,
                        help="Number of epochs for training. Default: 200")
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
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, threshold=0,
                                           verbose=True, min_lr=1e-03)
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
        running_dice = 0.0

        for images, masks in self.dataloaders[phase]:
            loss, bce_loss, dice_loss, outputs = self.forward(images, masks)
            if phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            running_bce_loss += bce_loss.item()
            running_dice_loss += dice_loss.item()

            outputs = outputs.detach().cpu()
            running_dice += compute_dice(outputs, masks).item()

        epoch_loss = running_loss / len(self.dataloaders[phase])
        epoch_bce_loss = running_bce_loss / len(self.dataloaders[phase])
        epoch_dice_loss = running_dice_loss / len(self.dataloaders[phase])
        epoch_dice = running_dice / len(self.dataloaders[phase])

        self.loss[phase].append(epoch_loss)
        self.bce_loss[phase].append(epoch_bce_loss)
        self.dice_loss[phase].append(epoch_dice_loss)
        self.dice[phase].append(epoch_dice)

        torch.cuda.empty_cache()

        return epoch_loss, epoch_bce_loss, epoch_dice_loss, epoch_dice

    def plot_history(self):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
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

        ax4.plot(self.dice['train'], '-b', label='Training')
        ax4.plot(self.dice['valid'], '-r', label='Validation')
        ax4.set_title("Dice", fontweight='bold')
        ax4.legend(loc="upper right", frameon=False)

        output_filename = os.path.join(self.training_history_path,
                                       "{}_fold_{}.pdf".format(self.model_save_name, self.fold))
        fig.tight_layout()
        fig.savefig(output_filename)

    def write_history(self):
        output_filename = os.path.join(self.training_history_path,
                                       "{}_fold_{}.txt".format(self.model_save_name, self.fold))
        header = ["Training loss", "Validation loss",
                  "Training bce loss", "Validation loss",
                  "Training dice loss", "Validation dice loss",
                  "Training dice", "Validation dice"]

        with open(output_filename, "a") as f:
            f.write("\t".join(header) + "\n")
            for i in range(self.num_epochs):
                res = [self.loss['train'][i], self.loss['valid'][i],
                       self.bce_loss['train'][i], self.bce_loss['valid'][i],
                       self.dice_loss['train'][i], self.dice_loss['valid'][i],
                       self.dice['train'][i], self.dice['valid'][i]]

                f.write("\t".join(map(str, res)) + "\n")

    def start(self):
        print("Train on {} mini-batches, validate on {} mini-batches".format(len(self.dataloaders["train"]),
                                                                             len(self.dataloaders["valid"])))

        for epoch in range(self.num_epochs):
            start = time.strftime("%D-%H:%M:%S")
            print("Epoch: {}/{} |  time : {}".format(epoch + 1, self.num_epochs, start))
            print("=================================================================")

            train_loss, train_bce_loss, train_dice_loss, train_dice = self.iterate("train")
            with torch.no_grad():
                valid_loss, valid_bce_loss, valid_dice_loss, valid_dice = self.iterate("valid")

            print("train_loss: %0.8f, train_bce_loss: %0.8f, train_dice_loss: %0.8f, train_dice: %0.8f" % (
                train_loss, train_bce_loss, train_dice_loss, train_dice))
            print("valid_loss: %0.8f, valid_bce_loss: %0.8f, valid_dice_loss: %0.8f, valid_dice: %0.8f" % (
                valid_loss, valid_bce_loss, valid_dice_loss, valid_dice))

            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.model.state_dict(),
            }

            self.scheduler.step(metrics=valid_loss)
            if valid_loss < self.best_loss:
                print("******** Validation loss improved from {} to {}, saving state ********".format(self.best_loss,
                                                                                                      valid_loss))
                state["best_loss"] = self.best_loss = valid_loss
                filename = os.path.join(self.model_save_path, "{}_fold_{}.pt".format(self.model_save_name, self.fold))
                if os.path.exists(filename):
                    os.remove(filename)
                torch.save(state, filename)

            print()


def main():
    args = parse_args()

    seed_torch(seed=42)

    model = None
    if args.model == "UResNet34":
        model = UResNet34()
    elif args.model == "UResNet50":
        model = UResNet50()
    elif args.model == "UResNext50":
        model = UResNext50()

    model_save_path = os.path.join(SAVE_MODEL_PATH, args.model)
    training_history_path = os.path.join(TRAINING_HISTORY_PATH, args.model)

    model_trainer = Trainer(model=model,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            num_epochs=args.num_epochs,
                            model_save_path=model_save_path,
                            training_history_path=training_history_path,
                            model_save_name=args.model,
                            fold=args.fold)
    model_trainer.start()
    model_trainer.write_history()
    model_trainer.plot_history()


if __name__ == '__main__':
    main()
