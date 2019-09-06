import os
import pathlib

from configure import SAVE_MODEL_PATH, TRAINING_HISTORY_PATH

model_list = ['UResNet34']
fold_list = [0]

for model in model_list:
    model_save_path = os.path.join(SAVE_MODEL_PATH, model)
    if not os.path.exists(model_save_path):
        pathlib.Path(model_save_path).mkdir(parents=True, exist_ok=True)

    training_history_path = os.path.join(TRAINING_HISTORY_PATH, model)
    if not os.path.exists(training_history_path):
        pathlib.Path(training_history_path).mkdir(parents=True, exist_ok=True)

    for fold in fold_list:
        job_name = "{}_fold_{}".format(model, fold)
        command = "sbatch -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
                  "./cluster_err/" + job_name + "_err.txt -t 120:00:00 --mem 50G -A rwth0233 "
        command += "--partition=c18g -c 2 --gres=gpu:1 train.zsh"
        os.system(command + " " + model + " " + str(fold))
