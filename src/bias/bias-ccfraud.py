import pandas as pd
import numpy as np
import sklearn as sk
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from aif360.explainers import MetricTextExplainer
import random
import json
from process import predo, preres_cc, compute_metrics
import os

'''data preprocess'''
feature_size = 8


# 每个数据的变量名
mean_list = ['gender', 'state','cardholder','balance','numTrans',
             'numIntTrans','creditLine','fraudRisk']


# 原数据处理
# data中所有数据需要修改成数值格式
# todo age 和gender需要进一步划分成二分类？
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
# train_data = pd.read_csv('./bias_data/ccfraud_train.csv', sep=',', names=[i for i in range(feature_size)])
# train = pd.DataFrame(train_data)
# train.columns = mean_list

# test_data = pd.read_csv('./bias_data/ccfraud_test.csv', sep=',', names=[i for i in range(feature_size)])
# test = pd.DataFrame(test_data)
# test.columns = mean_list # 表格重新写表头

# # method结果读取
# # todo 标签需要转换适配各个数据集
# res, index = preres_cc(test.values.tolist(), os.path.join(current_dir, 'CALM', 'flare_ccfraud_desc_write_out_info.json'))
# res = pd.DataFrame(res)
# res.columns = mean_list

# res = res.drop(index)
# test = test.drop(index)

def prepare_input_data(filename, output_file = None):
    my_data = pd.read_csv(filename, sep=',', names=[i for i in range(feature_size)])
    my_data_df = pd.DataFrame(my_data)
    my_data_df.columns = mean_list
    if output_file:
        _, index_to_drop = preres_cc(my_data_df.values.tolist(), output_file)
        my_data_df = my_data_df.drop(index_to_drop)      
    return my_data_df

def prepare_output_data(output_filename, test_filename):
    my_test_df = prepare_input_data(test_filename)
    my_data, my_idx = preres_cc(my_test_df.values.tolist(), output_filename)
    my_data_df = pd.DataFrame(my_data)
    my_data_df.columns = mean_list
    if len(my_idx) > 0:
        my_data_df = my_data_df.drop(my_idx)
    return my_data_df

train = prepare_input_data(os.path.join(current_dir, "bias_data", "ccfraud_train.csv"))
test = prepare_input_data(os.path.join(current_dir, "bias_data", "ccfraud_test.csv"), output_file=os.path.join(current_dir, "CALM", "flare_ccfraud_desc_write_out_info.json"))
res = prepare_output_data(os.path.join(current_dir, "CALM", "flare_ccfraud_desc_write_out_info.json"), os.path.join(current_dir, "bias_data", "ccfraud_test.csv"))

'''data bias test'''
# 测试数据本身偏见性
# favorable_label 为好的数值，即无风险的代表数字
# unfavorable_label 为坏的数值
# df 为数据
# label_names 作为目标的变量名
# protected_attribute_names 需要保护的变量名，含偏见的变量名
def disparate_impact(input_df):
    input_data = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=input_df, label_names=['fraudRisk'], protected_attribute_names=['gender'])
    final_res = dict()
    
    # Gender DI
    metric = BinaryLabelDatasetMetric(input_data, unprivileged_groups=[{'gender':2}], privileged_groups=[{'gender':1}])
    # text_res = MetricTextExplainer(metric)        
    final_res['Gender'] = metric.disparate_impact()

    return final_res

def bias_test(output_df, input_test_df):
    llm_output_data = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=output_df, label_names=['fraudRisk'], protected_attribute_names=['gender'])
    input_test_data = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=input_test_df, label_names=['fraudRisk'], protected_attribute_names=['gender'])
    final_res = {'EOD': {}, "AOD": {}}
    
    # Gender EOD and AOD
    metric = ClassificationMetric(input_test_data, llm_output_data, unprivileged_groups=[{'gender':2}], privileged_groups=[{'gender':1}])
    # text_res = MetricTextExplainer(metric)        
    final_res['EOD']["Gender"] = metric.equal_opportunity_difference()
    final_res['AOD']["Gender"] = metric.average_odds_difference()

    return final_res

print("Train DI:", disparate_impact(train))
print("Train DI:", disparate_impact(test))
print("Bias Test:", bias_test(res, test))
print("Results:", compute_metrics(os.path.join(current_dir, "CALM", "flare_ccfraud_desc_write_out_info.json")))

# test_data = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=test, label_names=['fraudRisk'], protected_attribute_names=['gender'])

# # unprivileged_groups 弱势群体，例如{gender：1}表示弱势群体是女性，list[]内可以叠加，也可以多次使用分开算
# # privileged_groups 优势群体，例如{gender：2}表示优势群体是男性，
# metric = BinaryLabelDatasetMetric(test_data, unprivileged_groups=[{'gender':2}], privileged_groups=[{'gender':1}])
# text_res = MetricTextExplainer(metric)

# print('DI:', text_res.disparate_impact())


# train_data = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=train, label_names=['fraudRisk'], protected_attribute_names=['gender'])

# # unprivileged_groups 弱势群体，例如{gender：1}表示弱势群体是女性，list[]内可以叠加，也可以多次使用分开算
# # privileged_groups 优势群体，例如{gender：2}表示优势群体是男性，
# metric = BinaryLabelDatasetMetric(train_data, unprivileged_groups=[{'gender':2}], privileged_groups=[{'gender':1}])
# text_res = MetricTextExplainer(metric)

# print('DI:', text_res.disparate_impact())


'''method bias test'''
# 测试模型偏见性
# favorable_label 为好的数值，即无风险的代表数字
# unfavorable_label 为坏的数值
# df 为method输出的数据
# label_names 作为目标的变量名
# protected_attribute_names 需要保护的变量名，含偏见的变量名
# res_data = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=res, label_names=['fraudRisk'], protected_attribute_names=['gender'])

# metric = ClassificationMetric(test_data, res_data, unprivileged_groups=[{'gender':2}], privileged_groups=[{'gender':1}])
# text_res = MetricTextExplainer(metric)

# print('EOD:', text_res.equal_opportunity_difference())
# print('ERR:', text_res.average_odds_difference())

print('down')