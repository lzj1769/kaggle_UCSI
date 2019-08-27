import os
import pandas as pd
import pathlib
from sklearn.model_selection import StratifiedKFold

from configure import TRAIN_DF_PATH, SPLIT_FOLDER

if not os.path.exists(SPLIT_FOLDER):
    pathlib.Path(SPLIT_FOLDER).mkdir(parents=True, exist_ok=True)

df = pd.read_csv(TRAIN_DF_PATH)
df['ImageId'], df['Label'] = zip(*df['Image_Label'].str.split('_'))
# df['Label'] = df['Label'].astype(int)
df = df.pivot(index='ImageId', columns='Label', values='EncodedPixels')
df['count'] = df.notnull().sum(axis=1).astype(int)

skf = StratifiedKFold(n_splits=5, random_state=42)
X = df.index
y = df['count']
for i, (train_index, valid_index) in enumerate(skf.split(X=X, y=y)):
    df_train, df_valid = df.iloc[train_index], df.iloc[valid_index]
    df_train.to_csv(os.path.join(SPLIT_FOLDER, "fold_{}_train.csv".format(i)))
    df_valid.to_csv(os.path.join(SPLIT_FOLDER, "fold_{}_valid.csv".format(i)))
