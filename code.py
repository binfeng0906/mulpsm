import os

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import pandas as pd
import numpy as np
import math
import random
import shutil
from sklearn.model_selection import KFold

file_path = "../data/avm/avm.csv"
file_score_path = "../data/avm/avm_score_123.csv"
file_match_path = "../data/avm/avm_match_123_caliper_0.1.csv"
caliper = 0.1


def load_origin_data():
    data = pd.read_csv(file_path, encoding="gbk")
    data = data.loc[data["surgery"] != 5]
    data = data.loc[data["surgery"] != 4]
    return data

def load_data():
    data = load_origin_data()
    variables = ["Sex", "Age","Epilepsy", "No_of_ICH", "GCS", "2TD", "HBP","smoking", "drinking", "SPC", "hrIA"]
    x_data = data.loc[:, variables]
    y_data = data.loc[:, "surgery"]

    return x_data.values, y_data.values

def save_PSM(y_hat):
    data = load_origin_data()

    column_size = y_hat.shape[1]
    for i in range(column_size):
        column_name = "score_" + str(i) + "_index"
        data[column_name] = y_hat[:, i]

    data.to_csv(file_score_path, index=False, encoding="gbk")
    pass

def train():
    X, y = load_data()

    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    kf = KFold(n_splits=10, random_state=0, shuffle=True)

    max_score = -1
    for train,test in kf.split(X):
        X_train, X_test, y_train, y_test = X_scaled[train], X_scaled[test], y[train], y[test]
        clf = LogisticRegression(random_state=0).fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        if score > max_score:
            y_hat_prob = clf.predict_proba(X_scaled)
            max_score = score

    print("max_logist_regression:%f"%max_score)
    # y_hat_prob = clf.predict_proba(X_scaled)
    save_PSM(y_hat_prob)
    pass


def link_path_to_list(path:str):
    if path.find("_") == -1:
        return [path]
    items = path.split("_")
    return list(items)

def list_to_link_path(path:list):
    if len(path) == 0:
        return ""
    s = str(path[0])
    for item in range(1, len(path)):
        s = s + "_" + str(path[item])
    return s

def calc_distance(proba_a, prob_b):
    return np.linalg.norm(proba_a - prob_b)

def apply_calc_distance(row, *args):
    control_prob = args[0]
    score_size = args[1]
    treat_prob = row[-score_size:]

    return calc_distance(control_prob.values, treat_prob.values)

def find_match_index(data:pd.DataFrame, total_data:pd.DataFrame, control_values, score_size:int, distance_map:map, use_caliper = False):
    if data.shape[0] == 0:
        return distance_map

    new_distance_map = {}
    for link_list in distance_map.keys():
        distance = distance_map[link_list]

        distance_series = data.apply(lambda row: apply_calc_distance(row, control_values, score_size), axis=1)
        for index in distance_series.index:
            id = int(data.loc[index]["id"])
            cur_distance = distance_series[index]
            if use_caliper and cur_distance >= caliper:
                pass
            else:
                #id----link_list[0, -1]的距离合
                new_link_list = link_path_to_list(link_list)
                for before_id in new_link_list[0:-1]:
                    before_values = total_data[total_data["id"] == int(before_id)].values[0, -score_size:]
                    cur_values = total_data[total_data["id"] == int(id)].values[0, -score_size:]
                    cur_distance += calc_distance(before_values, cur_values)
                    pass
                new_link_list.append(id)
                new_distance_map[list_to_link_path(new_link_list)] = distance + cur_distance

    return new_distance_map


def match(seed = 0):
    data = pd.read_csv(file_score_path, encoding="gbk")

    data_surgery = data["surgery"]
    data_surgery = list(set(data_surgery))

    if len(data_surgery) <= 1:
        print("no valid data")
        return

    control_label = data_surgery[0]
    control_group = data[data["surgery"] == control_label]

    control_group_indexs = list(control_group.index)
    random.seed(seed)
    random.shuffle(control_group_indexs)

    used_id_list = []
    match_result = {}
    for index in control_group_indexs:
        control_values = control_group.loc[index][-len(data_surgery):]

        id = int(control_group.loc[index]["id"])
        used_id_list.append(id)


        link_list = [id]
        distance_map = {}
        distance_map[list_to_link_path(link_list)] = 0

        for treat_label_index in range(1, len(data_surgery)):
            treat_data = data.drop(data[data["id"].isin(used_id_list)].index)
            treat_data = treat_data[treat_data["surgery"] == data_surgery[treat_label_index]]

            distance_map = find_match_index(treat_data, data, control_values, len(data_surgery), distance_map, True)

            if len(distance_map) == 0:
                break

        if len(distance_map) == 0:
            match_result[id] = [-1 for i in range(1, len(data_surgery))]
            continue

        first_path = list(distance_map.keys())[0]
        if len(link_path_to_list(first_path)) != len(data_surgery):
            match_result[id] = [-1 for i in range(1, len(data_surgery))]
            continue

        #依据distance查找最小link
        min_distance = -1
        min_link = ""
        for link_path in distance_map.keys():
            cur_distance = distance_map[link_path]
            if min_distance == -1 or min_distance > cur_distance:
                min_distance = cur_distance
                min_link = link_path
            else:
                continue

        print(control_group_indexs.index(index)/control_group.shape[0])

        #存放link
        min_link = link_path_to_list(min_link)
        if len(min_link) != 0:
            min_link.remove(min_link[0])
        min_link = [int(item) for item in min_link] #str 转 int
        match_result[id] = min_link
        used_id_list = used_id_list + min_link

    print("process over")
    print(match_result)
    pd.DataFrame(match_result).to_csv(file_match_path, index=False, encoding="gbk")

    pass

def calc_continue_smd(treat_col, control_col):
    if math.sqrt((treat_col.var() + control_col.var()) / 2) == 0:
        return 0
    return abs(treat_col.mean() - control_col.mean()) / math.sqrt((treat_col.var() + control_col.var()) / 2)

def calc_smd(control_data:pd.DataFrame, treat_data:pd.DataFrame):
    columns = list(treat_data.columns)
    columns.remove("name")
    columns.remove("id")
    columns.remove("surgery")
    columns.remove("location")
    columns.remove("deep_region")

    result = {}
    for column in columns:
        result[column] = calc_continue_smd(treat_data[column], control_data[column])

    return result

def calc_group_smd(smd_list:list):
    smd_keys = smd_list[0].keys()
    smd_values = {}

    for key in smd_keys:
        value = 0
        for smd_map in smd_list:
            value += smd_map[key]
        value /= len(smd_list)
        smd_values[key] = value
    return smd_values

def save_result(save_folder_name:str, result:map):
    folder_path = os.path.join("../data/avm", save_folder_name)
    os.mkdir(folder_path)

    math_save_path = os.path.join(folder_path, "match.csv")
    shutil.copy(file_match_path, math_save_path)

    smd_save_path = os.path.join(folder_path, "smd.csv")
    pd.DataFrame(result, index=[0]).to_csv(smd_save_path, index=False, encoding="gbk")

def evaluate(save_folder_name):
    match_data = pd.read_csv(file_match_path, encoding="gbk")
    data = load_origin_data()

    group_size = match_data.shape[0]
    smd_list = []

    for group_index in range(0, group_size):
        treat_group_ids = match_data.iloc[group_index, :]
        treat_group_ids = treat_group_ids[treat_group_ids != -1]

        control_group_ids = treat_group_ids.index
        control_group_ids = {int(id) for id in control_group_ids}

        treat_group = data[data["id"].isin(treat_group_ids)]
        control_group = data[data["id"].isin(control_group_ids)]

        smd_map = calc_smd(control_group, treat_group)
        smd_list.append(smd_map)

    smd = calc_group_smd(smd_list)
    max_column_smd = -1
    for key in smd:
       if smd[key] >= max_column_smd:
           max_column_smd = smd[key]
       print("key:%s\t value:%f"%(key, smd[key]))
    save_folder_name = save_folder_name + "_" +str(max_column_smd)
    save_result(save_folder_name, smd)

    pass

def preprocess():
    data = pd.read_csv("../data/avm/AVM_PSMnew.csv", encoding="gbk", usecols=["No.", "姓名", "Sex", "Age",
                                                                "Epilepsy", "No_of_ICH", "GCS", "2TD", "HBP", "functional_region", "deep_region", "location",
                                                                "smoking", "drinking", "SPC", "hrIA", "Treatment"])
    data.rename(columns={"No.":"id", "姓名":"name", "Treatment":"surgery"}, inplace=True)

    locations = list(set(data["SPC"]))
    data["SPC"] = data.apply(lambda row: locations.index(row["SPC"]), axis=1)

    locations = list(set(data["location"]))
    data["location"] = data.apply(lambda row: locations.index(row["location"]), axis=1)

    data.to_csv(file_path, index=False, encoding="gbk")
    pass

if __name__ == '__main__':
    preprocess()
    train()
    for seed in range(0, 100):
        match(seed)
        evaluate(str(seed))
    pass
