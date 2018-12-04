# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 21:17:27 2018

@author: wzy
"""
import pandas as pd
from copy import deepcopy

NO_1 = pd.read_csv("lgb.csv")
NO_2 = pd.read_csv("xgb.csv")
NO_3 = pd.read_csv("catboost.csv")

NO_1_m = list(deepcopy(NO_1['price']))
NO_2_m = list(deepcopy(NO_2['price']))
NO_3_m = list(deepcopy(NO_3['price']))
ID = list(deepcopy(NO_1['id']))
result = []
# 比例按照线上比分计算出来
for i in range(len(NO_1)):
	 # 0.62/0.3/0.08
    # 0.63/0.25/0.12掉分
    # 0.62/0.31/0.07掉分
    # 0.62/0.29/0.09上分
    temp = NO_1_m[i] * 0.62 + NO_2_m[i] * (0.27) + NO_3_m[i] * (0.11)
    result.append(temp)

test_lgb = pd.DataFrame({'id': ID, 'price': result})
test_lgb.to_csv('new.csv', index=False)
