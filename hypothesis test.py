import numpy as np

import pandas as pd
from scipy.stats import wilcoxon
import os

tar_p = './seg_record/every_5'
real = pd.read_csv('./seg_record/2021_64_t1ce_real.csv')
path = os.listdir(tar_p)
for i in path:
    dif = pd.read_csv(os.path.join(tar_p, i))

    r51 = real['51'][:56]
    r51 = r51.replace(np.nan, 0)
    r102 = real['102'][:56]
    r102 = r102.replace(np.nan, 0)
    r204 = real['204'][:56]
    r204 = r204.replace(np.nan, 0)
    rm = (r51 + r102 + r204) / 3

    d51 = dif['51'][:56]
    d51 = d51.replace(np.nan, 0)
    d102 = dif['102'][:56]
    d102 = d102.replace(np.nan, 0)
    d204 = dif['204'][:56]
    d204 = d204.replace(np.nan, 0)
    dm = (d51 + d102 + d204) / 3

    dif51 = d51 - r51
    dif102 = d102 - r102
    dif204 = d204 - r204
    difmean = dm - rm

    # Perform the Wilcoxon signed-rank test
    statistic, pvalue51 = wilcoxon(dif51)
    statistic, pvalue102 = wilcoxon(dif102)
    statistic, pvalue204 = wilcoxon(dif204)
    statistic, pvaluem = wilcoxon(difmean)

    print(i)
    print('51 : {}   '.format(pvalue51) + '102 : {}   '.format(pvalue102) + '204 : {}   '.format(pvalue204))
    print('mean : {}   '.format(pvaluem))
    print('--------------------------------------------------------')
