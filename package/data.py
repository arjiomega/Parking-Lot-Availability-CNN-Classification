from pathlib import Path
import os
import sys

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from config import config




df = pd.read_csv(Path(config.DATA_DIR,'labels.csv'))
df.set_index('image_name',inplace=True)

file_names = os.listdir(Path(config.DATA_DIR,'training'))

# Split into their own folders
for file in file_names:

    label = df.loc[file]['label']

    if not os.path.exists(Path(config.DATA_DIR,label)):
        os.makedirs(Path(config.DATA_DIR,label))

    os.rename(Path(config.DATA_DIR,'training',file), Path(config.DATA_DIR,'training',label,file))

if __name__ == "__main__":
    ...