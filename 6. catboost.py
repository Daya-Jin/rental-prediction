# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:45:27 2018
@author: wzy
"""
import pandas as pd
import copy
import numpy as np
from math import isnan
import catboost as cb
from catboost import Pool

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 房屋朝向的汉字编码处理
train_vec = copy.deepcopy(train['房屋朝向'])
train_vec_dic = list(set(train_vec))
num_train_vec_dic = len(train_vec_dic)
encode_train_vec_dic = list(np.arange(num_train_vec_dic))
dic_enc_train_vec = dict(map(lambda x, y: [x, y], train_vec_dic, encode_train_vec_dic))
encode_train_vec = []
for i in train_vec:
    temp = dic_enc_train_vec[i]
    encode_train_vec.append(temp)
test_vec = copy.deepcopy(test['房屋朝向'])
test_vec_dic = list(set(test_vec))
te_vec_new = []
for i in test_vec_dic:
    if i not in train_vec_dic:
        te_vec_new.append(i)
num_te_vec_new = len(te_vec_new)
enc_te_vec_new = list(np.arange(num_train_vec_dic, num_train_vec_dic+num_te_vec_new))
dic_te_vec_new = dict(map(lambda x, y: [x, y], te_vec_new, enc_te_vec_new))
dic_te_vec = dict(dic_enc_train_vec, **dic_te_vec_new)
encode_test_vec = []
for i in test_vec:
    temp = dic_te_vec[i]
    encode_test_vec.append(temp)
train['房屋朝向1'] = encode_train_vec
test['房屋朝向1'] = encode_test_vec
del dic_enc_train_vec, dic_te_vec, dic_te_vec_new, enc_te_vec_new, encode_test_vec
del encode_train_vec, encode_train_vec_dic, i, num_te_vec_new, num_train_vec_dic
del te_vec_new, temp, test_vec, test_vec_dic, train_vec, train_vec_dic

train_vec = list(copy.deepcopy(train['房屋朝向']))
diract = []
for i in train_vec:
    temp = i.split()
    temp = temp[0]
    diract.append(temp)
Diract = []
for i in diract:
    if i == '南':
        temp = 0
    elif i == '东南':
        temp = 1
    elif i == '东':
        temp = 2
    elif i == '西南':
        temp = 3
    elif i == '北':
        temp = 4
    elif i == '西':
        temp = 5
    elif i == '东北':
        temp = 6
    elif i == '西北':
        temp = 7
    Diract.append(temp)
train.drop('房屋朝向',axis=1, inplace=True)
train['房屋朝向2'] = Diract
del train_vec, diract, i, temp, Diract
test_vec = list(copy.deepcopy(test['房屋朝向']))
diract = []
for i in test_vec:
    temp = i.split()
    temp = temp[0]
    diract.append(temp)
Diract = []
for i in diract:
    if i == '南':
        temp = 0
    elif i == '东南':
        temp = 1
    elif i == '东':
        temp = 2
    elif i == '西南':
        temp = 3
    elif i == '北':
        temp = 4
    elif i == '西':
        temp = 5
    elif i == '东北':
        temp = 6
    elif i == '西北':
        temp = 7
    Diract.append(temp)
test.drop('房屋朝向',axis=1, inplace=True)
test['房屋朝向2'] = Diract
del test_vec, diract, i, temp, Diract

train_subway = list(copy.deepcopy(train['地铁线路']))
test_subway = list(copy.deepcopy(test['地铁线路']))
Train_subway = []
for i in train_subway:
    if isnan(i):
        Train_subway.append(0)
    else:
        Train_subway.append(i)
Test_subway = []
for i in test_subway:
    if isnan(i):
        Test_subway.append(0)
    else:
        Test_subway.append(i)

train.drop('地铁线路',axis=1, inplace=True)
train['地铁线路'] = Train_subway
test.drop('地铁线路',axis=1, inplace=True)
test['地铁线路'] = Test_subway
del Test_subway, Train_subway, i, test_subway, train_subway

train_station = list(copy.deepcopy(train['地铁站点']))
test_station = list(copy.deepcopy(test['地铁站点']))
Train_station = []
for i in train_station:
    if isnan(i):
        Train_station.append(0)
    else:
        Train_station.append(i)
Test_station = []
for i in test_station:
    if isnan(i):
        Test_station.append(0)
    else:
        Test_station.append(i)
train.drop('地铁站点',axis=1, inplace=True)
train['地铁站点'] = Train_station
test.drop('地铁站点',axis=1, inplace=True)
test['地铁站点'] = Test_station
del Test_station, Train_station, i, test_station, train_station

train = train.sort_values(by=['小区名', '楼层', '时间'], ascending=(True, True, True))
train_num_house = list(copy.deepcopy(train['小区房屋出租数量']))
count = []
num = len(train_num_house)
for i in range(num):
    if isnan(train_num_house[i]):
        count.append(count[i-1])
    else:
        count.append(train_num_house[i])
train.drop('小区房屋出租数量',axis=1, inplace=True)
train['小区房屋出租数量'] = count
test = test.sort_values(by=['小区名', '楼层', '时间'], ascending=(True, True, True))
test_num_house = list(copy.deepcopy(test['小区房屋出租数量']))
count = []
num = len(test_num_house)
for i in range(num):
    if isnan(test_num_house[i]):
        count.append(count[i-1])
    else:
        count.append(test_num_house[i])
test.drop('小区房屋出租数量',axis=1, inplace=True)
test['小区房屋出租数量'] = count
del count, i, num, test_num_house, train_num_house

train_province = list(copy.deepcopy(train['区']))
test_province = list(copy.deepcopy(test['区']))
Train_province = []
for i in train_province:
    if isnan(i):
        Train_province.append(5)
    else:
        Train_province.append(i)
Test_province = []
for i in test_province:
    if isnan(i):
        Test_province.append(5)
    else:
        Test_province.append(i)
train.drop('区',axis=1, inplace=True)
train['区'] = Train_province
test.drop('区',axis=1, inplace=True)
test['区'] = Test_province
del Test_province, Train_province, i, test_province, train_province

train_location = list(copy.deepcopy(train['位置']))
test_location = list(copy.deepcopy(test['位置']))
Train_location = []
for i in train_location:
    if isnan(i):
        Train_location.append(76)
    else:
        Train_location.append(i)
Test_location = []
for i in test_location:
    if isnan(i):
        Test_location.append(76)
    else:
        Test_location.append(i)
train.drop('位置',axis=1, inplace=True)
train['位置'] = Train_location
test.drop('位置',axis=1, inplace=True)
test['位置'] = Test_location
del Test_location, Train_location, i, test_location, train_location

train_sub = []
train_sub = pd.DataFrame(train_sub)
test_sub = []
test_sub = pd.DataFrame(test_sub)
train_sub['小区名'] = list(copy.deepcopy(train['小区名']))
train_sub['距离'] = list(copy.deepcopy(train['距离']))
train_sub['地铁线路'] = list(copy.deepcopy(train['地铁线路']))
train_sub['地铁站点'] = list(copy.deepcopy(train['地铁站点']))
test_sub['小区名'] = list(copy.deepcopy(test['小区名']))
test_sub['距离'] = list(copy.deepcopy(test['距离']))
test_sub['地铁线路'] = list(copy.deepcopy(test['地铁线路']))
test_sub['地铁站点'] = list(copy.deepcopy(test['地铁站点']))
train_sub = pd.concat([train_sub, test_sub], ignore_index=True)
train_sub = train_sub.drop_duplicates()
train_sub = train_sub.sort_values(by=['小区名'], ascending=(True))
train_sub = train_sub.dropna(axis=0, how='any')
del test_sub
train_sub_name = train_sub['小区名']
train_sub_dis = train_sub['距离']
dic_name_dis = dict(map(lambda x, y: [x, y], train_sub_name, train_sub_dis))
train_distance = list(copy.deepcopy(train['距离']))
train_name = list(copy.deepcopy(train['小区名']))
num = len(train_name)
distance = []
for i in range(num):
    if (train_name[i] in dic_name_dis.keys()) and (isnan(train_distance[i])):
        distance.append(dic_name_dis[train_name[i]])
    else:
        distance.append(train_distance[i])
train.drop('距离',axis=1, inplace=True)
train['距离'] = distance
test_distance = list(copy.deepcopy(test['距离']))
test_name = list(copy.deepcopy(test['小区名']))
num = len(test_name)
distance = []
for i in range(num):
    if (test_name[i] in dic_name_dis.keys()) and (isnan(test_distance[i])):
        distance.append(dic_name_dis[test_name[i]])
    else:
        distance.append(test_distance[i])
test.drop('距离',axis=1, inplace=True)
test['距离'] = distance
del dic_name_dis, distance, i, num, test_distance, test_name, train_distance
del train_name, train_sub_dis, train_sub_name

train_sub_name = train_sub['小区名']
train_sub_subway = train_sub['地铁线路']
dic_name_subway = dict(map(lambda x, y: [x, y], train_sub_name, train_sub_subway))
train_name = list(copy.deepcopy(train['小区名']))
train_subway = list(copy.deepcopy(train['地铁线路']))
num = len(train_name)
subway = []
for i in range(num):
    if (train_name[i] in dic_name_subway.keys()) and (train_subway[i]==0):
        subway.append(dic_name_subway[train_name[i]])
    else:
        subway.append(train_subway[i])
train.drop('地铁线路',axis=1, inplace=True)
train['地铁线路'] = subway
test_name = list(copy.deepcopy(test['小区名']))
test_subway = list(copy.deepcopy(test['地铁线路']))
num = len(test_name)
subway = []
for i in range(num):
    if (test_name[i] in dic_name_subway.keys()) and (test_subway[i]==0):
        subway.append(dic_name_subway[test_name[i]])
    else:
        subway.append(test_subway[i])
test.drop('地铁线路',axis=1, inplace=True)
test['地铁线路'] = subway
del dic_name_subway, i, num, subway, test_name, test_subway, train_name
del train_sub_name, train_sub_subway, train_subway

train_sub_name = train_sub['小区名']
train_sub_position = train_sub['地铁站点']
dic_name_position = dict(map(lambda x, y: [x, y], train_sub_name, train_sub_position))
train_name = list(copy.deepcopy(train['小区名']))
train_position = list(copy.deepcopy(train['地铁站点']))
num = len(train_name)
position = []
for i in range(num):
    if (train_name[i] in dic_name_position.keys()) and (train_position[i]==0):
        position.append(dic_name_position[train_name[i]])
    else:
        position.append(train_position[i])
train.drop('地铁站点',axis=1, inplace=True)
train['地铁站点'] = position
test_name = list(copy.deepcopy(test['小区名']))
test_position = list(copy.deepcopy(test['地铁站点']))
num = len(test_name)
position = []
for i in range(num):
    if (test_name[i] in dic_name_position.keys()) and (test_position[i]==0):
        position.append(dic_name_position[test_name[i]])
    else:
        position.append(test_position[i])
test.drop('地铁站点',axis=1, inplace=True)
test['地铁站点'] = position
del dic_name_position, i, num, position, test_name, test_position, train_name
del train_position, train_sub_name, train_sub_position, train_sub

train_distance = list(copy.deepcopy(train['距离']))
distance = []
for i in train_distance:
    if isnan(i):
        distance.append(0)
    else:
        distance.append(i)
train.drop('距离',axis=1, inplace=True)
train['距离'] = distance
test_distance = list(copy.deepcopy(test['距离']))
distance = []
for i in test_distance:
    if isnan(i):
        distance.append(0)
    else:
        distance.append(i)
test.drop('距离',axis=1, inplace=True)
test['距离'] = distance
del distance, i, test_distance, train_distance
#############################################################
## 修改日期：2018年11月2日
## 修改人员：wzy
#############################################################
train_live = list(copy.deepcopy(train['居住状态']))
live = []
for i in train_live:
    if isnan(i):
        live.append(0)
    else:
        live.append(i)
train.drop('居住状态',axis=1, inplace=True)
train['居住状态'] = live
test_live = list(copy.deepcopy(test['居住状态']))
live = []
for i in test_live:
    if isnan(i):
        live.append(0)
    else:
        live.append(i)
test.drop('居住状态',axis=1, inplace=True)
test['居住状态'] = live
del i, live, test_live, train_live

train_zx = list(copy.deepcopy(train['装修情况']))
zx = []
for i in train_zx:
    if isnan(i):
        zx.append(0)
    else:
        zx.append(i)
train.drop('装修情况',axis=1, inplace=True)
train['装修情况'] = zx
test_zx = list(copy.deepcopy(test['装修情况']))
zx = []
for i in test_zx:
    if isnan(i):
        zx.append(0)
    else:
        zx.append(i)
test.drop('装修情况',axis=1, inplace=True)
test['装修情况'] = zx
del i, test_zx, train_zx, zx

train_cz = list(copy.deepcopy(train['出租方式']))
cz = []
for i in train_cz:
    if isnan(i):
        cz.append(2)
    else:
        cz.append(i)
train.drop('出租方式',axis=1, inplace=True)
train['出租方式'] = cz
test_cz = list(copy.deepcopy(test['出租方式']))
cz = []
for i in test_cz:
    if isnan(i):
        cz.append(2)
    else:
        cz.append(i)
test.drop('出租方式',axis=1, inplace=True)
test['出租方式'] = cz
del cz, i, test_cz, train_cz

train = train.drop(train[train['房屋面积'] > 0.146].index)

train_ws = list(copy.deepcopy(train['卧室数量']))
train_t = list(copy.deepcopy(train['厅的数量']))
train_w = list(copy.deepcopy(train['卫的数量']))
total = []
num = len(train_ws)
for i in range(num):
    temp = train_ws[i] + train_t[i] + train_w[i]
    total.append(temp)
train['总房间数'] = total
test_ws = list(copy.deepcopy(test['卧室数量']))
test_t = list(copy.deepcopy(test['厅的数量']))
test_w = list(copy.deepcopy(test['卫的数量']))
total = []
num = len(test_ws)
for i in range(num):
    temp = test_ws[i] + test_t[i] + test_w[i]
    total.append(temp)
test['总房间数'] = total
del i, num, temp, test_t, test_w, test_ws, train_t, train_w, train_ws, total

train_ws = list(copy.deepcopy(train['卧室数量']))
train_w = list(copy.deepcopy(train['卫的数量']))
total = []
num = len(train_ws)
for i in range(num):
    temp = train_ws[i] + train_w[i]
    total.append(temp)
train['卧室和卫生间'] = total
test_ws = list(copy.deepcopy(test['卧室数量']))
test_w = list(copy.deepcopy(test['卫的数量']))
total = []
num = len(test_ws)
for i in range(num):
    temp = test_ws[i] + test_w[i]
    total.append(temp)
test['卧室和卫生间'] = total
del i, num, temp, test_w, test_ws, train_w, train_ws, total

train_acer = list(copy.deepcopy(train['房屋面积']))
acer = []
for i in train_acer:
    if i == 0:
        acer.append(0.0001)
    else:
        acer.append(i)
train.drop('房屋面积',axis=1, inplace=True)
train['房屋面积'] = acer
test_acer = list(copy.deepcopy(test['房屋面积']))
acer = []
for i in test_acer:
    if i == 0:
        acer.append(0.0001)
    else:
        acer.append(i)
test.drop('房屋面积',axis=1, inplace=True)
test['房屋面积'] = acer
del acer, i, test_acer, train_acer

#############################################################
## 修改日期：2018年11月3日
## 修改人员：wzy
#############################################################
# 这个特征保留
# 卧室占整个房间的比例
bed_sub_all = 0.3
train_sq = list(copy.deepcopy(train['房屋面积']))
train_ws = list(copy.deepcopy(train['卧室数量']))
ws_sq = []
num = len(train_sq)
for i in range(num):
    temp = (train_sq[i] * 10000 * bed_sub_all) / (train_ws[i] + 1)
    ws_sq.append(temp)
train['卧室均面积'] = ws_sq
test_sq = list(copy.deepcopy(test['房屋面积']))
test_ws = list(copy.deepcopy(test['卧室数量']))
ws_sq = []
num = len(test_sq)
for i in range(num):
    temp = (test_sq[i] * 10000 * bed_sub_all) / (test_ws[i] + 1)
    ws_sq.append(temp)
test['卧室均面积'] = ws_sq
del train_sq, train_ws, ws_sq, i, temp, num, test_sq, test_ws, bed_sub_all

# 这个特征保留
train_sq = list(copy.deepcopy(train['房屋面积']))
train_w = list(copy.deepcopy(train['卫的数量']))
w_sq = []
num = len(train_sq)
for i in range(num):
    temp = (train_sq[i] * 10000 / 14) / (train_w[i] + 1)
    w_sq.append(temp)
train['卫的均面积'] = w_sq
test_sq = list(copy.deepcopy(test['房屋面积']))
test_w = list(copy.deepcopy(test['卫的数量']))
w_sq = []
num = len(test_sq)
for i in range(num):
    temp = (test_sq[i] * 10000 / 14) / (test_w[i] + 1)
    w_sq.append(temp)
test['卫的均面积'] = w_sq
del train_sq, train_w, w_sq, temp, i, num, test_sq, test_w

train_wss = list(copy.deepcopy(train['卧室均面积']))
train_ws = list(copy.deepcopy(train['卧室数量']))
ws_s = []
num = len(train_wss)
for i in range(num):
    temp = train_wss[i] * train_ws[i]
    ws_s.append(temp)
train['卧室总面积'] = ws_s
test_wss = list(copy.deepcopy(test['卧室均面积']))
test_ws = list(copy.deepcopy(test['卧室数量']))
ws_s = []
num = len(test_wss)
for i in range(num):
    temp = test_wss[i] * test_ws[i]
    ws_s.append(temp)
test['卧室总面积'] = ws_s
del train_wss, train_ws, ws_s, num, i, temp, test_wss, test_ws

######################################################
## 1.98130后的特征测试
######################################################
train_acr_total = list(copy.deepcopy(train['房屋面积']))
train_bedroom_total = list(copy.deepcopy(train['卧室总面积']))
else_acer = []
num = len(train_acr_total)
for i in range(num):
    temp = (train_acr_total[i] * 10000) - train_bedroom_total[i]
    else_acer.append(temp)
train['除卧室外剩余面积'] = else_acer
test_acr_total = list(copy.deepcopy(test['房屋面积']))
test_bedroom_total = list(copy.deepcopy(test['卧室总面积']))
else_acer = []
num = len(test_acr_total)
for i in range(num):
    temp = (test_acr_total[i] * 10000) - test_bedroom_total[i]
    else_acer.append(temp)
test['除卧室外剩余面积'] = else_acer
del train_acr_total, train_bedroom_total, else_acer, num, i, temp, test_acr_total, test_bedroom_total

total_floor = list(copy.deepcopy(train['总楼层']))
Total_floor = []
for i in total_floor:
    if i == 0:
        Total_floor.append(0.01)
    else:
        Total_floor.append(i)
train.drop('总楼层',axis=1, inplace=True)
train['总楼层'] = Total_floor
total_floor = list(copy.deepcopy(test['总楼层']))
Total_floor = []
for i in total_floor:
    if i == 0:
        Total_floor.append(0.01)
    else:
        Total_floor.append(i)
test.drop('总楼层',axis=1, inplace=True)
test['总楼层'] = Total_floor
del Total_floor, i, total_floor

total_floor = list(copy.deepcopy(train['总楼层']))
floor = list(copy.deepcopy(train['楼层']))
num = len(total_floor)
Floor = []
for i in range(num):
    temp = total_floor[i] * 100 * (floor[i] + 1)
    Floor.append(temp)
train['具体楼层'] = Floor
total_floor = list(copy.deepcopy(test['总楼层']))
floor = list(copy.deepcopy(test['楼层']))
num = len(total_floor)
Floor = []
for i in range(num):
    temp = total_floor[i] * 100 * (floor[i] + 1)
    Floor.append(temp)
test['具体楼层'] = Floor
del total_floor, floor, Floor, num, i, temp

######################################################
## 1.94686后的特征测试 前面对房屋朝向做了重新处理
######################################################
# 上分区
train.drop('时间',axis=1, inplace=True)
test.drop('时间',axis=1, inplace=True)

# 对区按照租金平均数进行重新编码
train_qu = list(copy.deepcopy(train['区']))
train_label = list(copy.deepcopy(train['月租金']))
num = len(train_qu)
class_num = len(list(set(train_qu)))
qu_list = []
qu_money = []
for i in range(class_num):
    qu_list = []
    for j in range(num):
        if i == train_qu[j]:
            qu_list.append(train_label[j])
    qu_money.append(qu_list)
AVG = []
for i in qu_money:
    i = np.array(i)
    avg = np.mean(i)
    AVG.append(avg)
avg = list(copy.deepcopy(AVG))
avg = np.array(avg)
avg = np.sort(avg)
code = []
for i in range(class_num):
    for j in range(class_num):
        if AVG[i] == avg[j]:
            temp = j
            code.append(temp)
qu_code = []
for i in train_qu:
    i = int(i)
    qu_code.append(code[i])
train.drop('区',axis=1, inplace=True)
train['区'] = qu_code
test_qu = list(copy.deepcopy(test['区']))
qu_code = []
for i in test_qu:
    i = int(i)
    qu_code.append(code[i])
test.drop('区',axis=1, inplace=True)
test['区'] = qu_code
del train_qu, train_label, num, class_num, qu_list, qu_money, i, j, AVG, avg, code, qu_code, test_qu



train_location = list(copy.deepcopy(train['位置']))
train_label = list(copy.deepcopy(train['月租金']))
# 修改了前面代码，位置的缺失用76补齐
num_loc_class = len(list(set(train_location)))      # 153种位置 最小为0， 最大为153  76缺失
num_loc_samp = len(train_location)
loc_list = []
loc_money = []
for i in range(num_loc_class):
    loc_list = []
    for j in range(num_loc_samp):
        if i == train_location[j]:
            loc_list.append(train_label[j])
    loc_money.append(loc_list)
AVG = []
for i in loc_money:
    i = np.array(i)
    avg = np.mean(i)
    AVG.append(avg)
avg = list(copy.deepcopy(AVG))
avg = np.array(avg)
avg = np.sort(avg)
code = []
for i in range(num_loc_class):
    for j in range(num_loc_class):
        if AVG[i] == avg[j]:
            temp = j
            code.append(temp)
loc_code = []
for i in train_location:
    i = int(i)
    loc_code.append(code[i])
# train.drop('位置',axis=1, inplace=True)
train['位置rank'] = loc_code
test_location = list(copy.deepcopy(test['位置']))
loc_code = []
for i in test_location:
    i = int(i)
    loc_code.append(code[i])
# test.drop('位置',axis=1, inplace=True)
test['位置rank'] = loc_code
del train_location, train_label, num_loc_class, num_loc_samp, i, j, AVG, avg
del loc_code, code, test_location


train_location = list(copy.deepcopy(train['地铁线路']))
train_label = list(copy.deepcopy(train['月租金']))
# 修改了前面代码，位置的缺失用76补齐
num_loc_class = len(list(set(train_location)))      # 153种位置 最小为0， 最大为153  76缺失
num_loc_samp = len(train_location)
loc_list = []
loc_money = []
for i in range(num_loc_class):
    loc_list = []
    for j in range(num_loc_samp):
        if i == train_location[j]:
            loc_list.append(train_label[j])
    loc_money.append(loc_list)
AVG = []
for i in loc_money:
    i = np.array(i)
    avg = np.mean(i)
    AVG.append(avg)
avg = list(copy.deepcopy(AVG))
avg = np.array(avg)
avg = np.sort(avg)
code = []
for i in range(num_loc_class):
    for j in range(num_loc_class):
        if AVG[i] == avg[j]:
            temp = j
            code.append(temp)
loc_code = []
for i in train_location:
    i = int(i)
    loc_code.append(code[i])
# train.drop('地铁线路',axis=1, inplace=True)
train['地铁线路rank'] = loc_code
test_location = list(copy.deepcopy(test['地铁线路']))
loc_code = []
for i in test_location:
    i = int(i)
    loc_code.append(code[i])
# test.drop('地铁线路',axis=1, inplace=True)
test['地铁线路rank'] = loc_code
del train_location, train_label, num_loc_class, num_loc_samp, i, j, AVG, avg
del loc_code, code, test_location


######################################################
## 1.92593后的特征测试
######################################################
#地铁站点、小区名称进行rank
train = train.sort_values(by=['地铁站点'], ascending=(True))
train_name = list(copy.deepcopy(train['地铁站点']))
train_name_dic = list(set(train_name))
dic_num = len(train_name_dic)
code = list(np.arange(dic_num))
dic_enc_train_name = dict(map(lambda x, y: [x, y], train_name_dic, code))
encode_train_name = []
for i in train_name:
    temp = dic_enc_train_name[i]
    encode_train_name.append(temp)
test_name = list(copy.deepcopy(test['地铁站点']))
encode_test_name = []
for i in test_name:
    if i not in dic_enc_train_name.keys():
        print(i)
    else:
        temp = dic_enc_train_name[i]
    encode_test_name.append(temp)
train.drop('地铁站点',axis=1, inplace=True)
train['地铁站点'] = encode_train_name
test.drop('地铁站点',axis=1, inplace=True)
test['地铁站点'] = encode_test_name
del code, dic_enc_train_name, dic_num, encode_test_name, encode_train_name
del i, loc_list, loc_money, temp, test_name, train_name, train_name_dic

train_location = list(copy.deepcopy(train['地铁站点']))
train_label = list(copy.deepcopy(train['月租金']))
# 修改了前面代码，位置的缺失用76补齐
num_loc_class = len(list(set(train_location)))      # 153种位置 最小为0， 最大为153  76缺失
num_loc_samp = len(train_location)
loc_list = []
loc_money = []
for i in range(num_loc_class):
    loc_list = []
    for j in range(num_loc_samp):
        if i == train_location[j]:
            loc_list.append(train_label[j])
    loc_money.append(loc_list)
AVG = []
for i in loc_money:
    i = np.array(i)
    avg = np.mean(i)
    AVG.append(avg)
avg = list(copy.deepcopy(AVG))
avg = np.array(avg)
avg = np.sort(avg)
code = []
for i in range(num_loc_class):
    for j in range(num_loc_class):
        if AVG[i] == avg[j]:
            temp = j
            code.append(temp)
loc_code = []
for i in train_location:
    i = int(i)
    loc_code.append(code[i])
# train.drop('地铁站点',axis=1, inplace=True)
train['地铁站点rank'] = loc_code
test_location = list(copy.deepcopy(test['地铁站点']))
loc_code = []
for i in test_location:
    i = int(i)
    loc_code.append(code[i])
# test.drop('地铁站点',axis=1, inplace=True)
test['地铁站点rank'] = loc_code
del train_location, train_label, num_loc_class, num_loc_samp, i, j, AVG, avg
del loc_code, code, test_location

######################################################
## 1.92306后的特征测试
######################################################
# 房屋朝向2不可以做rank处理
# 修改卧室占整个房间的比例为0.3 原来为1/3
train_location = list(copy.deepcopy(train['装修情况']))
train_label = list(copy.deepcopy(train['月租金']))
# 修改了前面代码，位置的缺失用76补齐
num_loc_class = len(list(set(train_location)))      # 153种位置 最小为0， 最大为153  76缺失
num_loc_samp = len(train_location)
loc_list = []
loc_money = []
for i in range(num_loc_class):
    loc_list = []
    for j in range(num_loc_samp):
        if i == train_location[j]:
            loc_list.append(train_label[j])
    loc_money.append(loc_list)
AVG = []
for i in loc_money:
    i = np.array(i)
    avg = np.mean(i)
    AVG.append(avg)
avg = list(copy.deepcopy(AVG))
avg = np.array(avg)
avg = np.sort(avg)
code = []
for i in range(num_loc_class):
    for j in range(num_loc_class):
        if AVG[i] == avg[j]:
            temp = j
            code.append(temp)
loc_code = []
for i in train_location:
    i = int(i)
    loc_code.append(code[i])
train['装修情况rank'] = loc_code
test_location = list(copy.deepcopy(test['装修情况']))
loc_code = []
for i in test_location:
    i = int(i)
    loc_code.append(code[i])
test['装修情况rank'] = loc_code
del train_location, train_label, num_loc_class, num_loc_samp, i, j, AVG, avg
del loc_code, code, test_location

train_house_floor = list(copy.deepcopy(train['具体楼层']))
train_square = list(copy.deepcopy(train['房屋面积']))
total_square = []
for i in range(len(train_house_floor)):
    temp = train_house_floor[i] * train_square[i]
    total_square.append(temp)
train['这层楼的房屋面积'] = total_square
del train_house_floor, train_square, i, temp, total_square
test_house_floor = list(copy.deepcopy(test['具体楼层']))
test_square = list(copy.deepcopy(test['房屋面积']))
total_square = []
for i in range(len(test_house_floor)):
    temp = test_house_floor[i] * test_square[i]
    total_square.append(temp)
test['这层楼的房屋面积'] = total_square
del test_house_floor, test_square, i, temp, total_square

######################################################
## 1.90010后的特征测试
######################################################
train_house_floor = list(copy.deepcopy(train['具体楼层']))
train_bedroom_total = list(copy.deepcopy(train['卧室总面积']))
bedroom_square = []
for i in range(len(train_house_floor)):
    temp = train_house_floor[i] * train_bedroom_total[i]
    bedroom_square.append(temp)
train['这层楼的卧室总面积'] = bedroom_square
del train_house_floor, train_bedroom_total, bedroom_square, i, temp
test_house_floor = list(copy.deepcopy(test['具体楼层']))
test_bedroom_total = list(copy.deepcopy(test['卧室总面积']))
bedroom_square = []
for i in range(len(test_house_floor)):
    temp = test_house_floor[i] * test_bedroom_total[i]
    bedroom_square.append(temp)
test['这层楼的卧室总面积'] = bedroom_square
del test_house_floor, test_bedroom_total, bedroom_square, i, temp

######################################################
## 1.88556后的特征测试
######################################################
train_badroom = list(copy.deepcopy(train['厅的数量']))
train_livingroom = list(copy.deepcopy(train['卧室数量']))
bad_and_living = []
for i in range(len(train_badroom)):
    temp = train_badroom[i] + train_livingroom[i]
    bad_and_living.append(temp)
train['卧室和厅'] = bad_and_living
test_badroom = list(copy.deepcopy(test['厅的数量']))
test_livingroom = list(copy.deepcopy(test['卧室数量']))
bad_and_living = []
for i in range(len(test_badroom)):
    temp = test_badroom[i] + test_livingroom[i]
    bad_and_living.append(temp)
test['卧室和厅'] = bad_and_living
del train_badroom, train_livingroom, bad_and_living, i, temp, test_badroom, test_livingroom

living_sub_all = 0.26
train_sq = list(copy.deepcopy(train['房屋面积']))
train_living = list(copy.deepcopy(train['厅的数量']))
ws_sq = []
num = len(train_sq)
for i in range(num):
    temp = (train_sq[i] * 10000 * living_sub_all) / (train_living[i] + 1)
    ws_sq.append(temp)
train['客厅均面积'] = ws_sq
test_sq = list(copy.deepcopy(test['房屋面积']))
test_living = list(copy.deepcopy(test['厅的数量']))
ws_sq = []
num = len(test_sq)
for i in range(num):
    temp = (test_sq[i] * 10000 * living_sub_all) / (test_living[i] + 1)
    ws_sq.append(temp)
test['客厅均面积'] = ws_sq
del train_sq, train_living, ws_sq, i, temp, num, test_sq, test_living, living_sub_all

train_wss = list(copy.deepcopy(train['客厅均面积']))
train_ws = list(copy.deepcopy(train['厅的数量']))
ws_s = []
num = len(train_wss)
for i in range(num):
    temp = train_wss[i] * train_ws[i]
    ws_s.append(temp)
train['客厅总面积'] = ws_s
test_wss = list(copy.deepcopy(test['客厅均面积']))
test_ws = list(copy.deepcopy(test['厅的数量']))
ws_s = []
num = len(test_wss)
for i in range(num):
    temp = test_wss[i] * test_ws[i]
    ws_s.append(temp)
test['客厅总面积'] = ws_s
del train_wss, train_ws, ws_s, num, i, temp, test_wss, test_ws

test = test.sort_values(by=['id'], ascending=(True))
test_id = list(copy.deepcopy(test['id']))
test.drop('id',axis=1, inplace=True)
train_label = list(copy.deepcopy(train['月租金']))
train.drop('月租金',axis=1, inplace=True)

train_pool = Pool(train, train_label, cat_features=None)
test_pool = Pool(test, cat_features=None)
cb_model = cb.CatBoostRegressor(depth=11, learning_rate=0.11, iterations=2729, l2_leaf_reg=0.1, model_size_reg=2, loss_function='RMSE')
cb_model.fit(train_pool, verbose=True)
preds = cb_model.predict(test_pool)

test_lgb = pd.DataFrame({'id': test_id, 'price': preds})
test_lgb.to_csv('./result/catboost.csv', index=False)
