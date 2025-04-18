import os 
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_injections = 'list_mdc1.txt'

injections = pd.read_csv(file_injections, sep=' ')

import corner
drops = ['#', 't0', 'tc', 'type', 'snrCE_Opt', 'lambda1', 'lambda2']

BNS = injections[injections['type']==1]
NSBH = injections[injections['type']==2]
BBH = injections[injections['type']==3]


BNS = BNS.drop(drops, axis=1)
NSBH = NSBH.drop(drops, axis=1)
BBH = BBH.drop(drops, axis=1)

f = corner.corner(BBH.to_dict(orient='list'))
plt.show()
