import os
import pandas as pd
import pathlib
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from configure import TRAIN_DF_PATH, SPLIT_FOLDER

if not os.path.exists(SPLIT_FOLDER):
    pathlib.Path(SPLIT_FOLDER).mkdir(parents=True, exist_ok=True)

df = pd.read_csv(TRAIN_DF_PATH)
df['ImageId'], df['Label'] = zip(*df['Image_Label'].str.split('_'))
df = df.pivot(index='ImageId', columns='Label', values='EncodedPixels')

df['isFish'] = 1 - df['Fish'].isnull().astype(int)
df['isFlower'] = 1 - df['Flower'].isnull().astype(int)
df['isGravel'] = 1 - df['Gravel'].isnull().astype(int)
df['isSugar'] = 1 - df['Sugar'].isnull().astype(int)

mskf = MultilabelStratifiedKFold(n_splits=5, random_state=42)

X = df.index
y1 = df['isFish'].values.tolist()
y2 = df['isFlower'].values.tolist()
y3 = df['isGravel'].values.tolist()
y4 = df['isSugar'].values.tolist()
y = list()
for i in range(len(y1)):
    y.append([y1[i], y2[i], y3[i], y4[i]])

for i, (train_index, valid_index) in enumerate(mskf.split(X=X, y=y)):
    df_train, df_valid = df.iloc[train_index], df.iloc[valid_index]
    df_train.to_csv(os.path.join(SPLIT_FOLDER, "fold_{}_train.csv".format(i)))
    df_valid.to_csv(os.path.join(SPLIT_FOLDER, "fold_{}_valid.csv".format(i)))
