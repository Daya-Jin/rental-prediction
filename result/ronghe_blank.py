# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 21:35:28 2018

@author: wzy
"""
import pandas as pd
from copy import deepcopy
from math import isnan

hu = pd.read_csv('statistic_pred.csv')
wu = pd.read_csv('new.csv')

ID = list(deepcopy(wu['id']))
hu_price = list(deepcopy(hu['Rental']))
wu_price = list(deepcopy(wu['price']))

result = []
for i in range(len(hu_price)):
    if isnan(hu_price[i]):
        result.append(wu_price[i])
    else:
        result.append(hu_price[i])
test_lgb = pd.DataFrame({'id': ID, 'price': result})
test_lgb.to_csv('final.csv', index=False)
