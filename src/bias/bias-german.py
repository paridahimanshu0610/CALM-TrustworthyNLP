import pandas as pd
import numpy as np
import sklearn as sk
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from aif360.explainers import MetricTextExplainer
import random
import json
import os
from process import predo, preres, compute_metrics

'''data preprocess'''
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
# '/Users/himanshu/Documents/Projects/CALM-TrustworthyNLP/data/credit_scoring/German/german.data' # "german.data"
feature_size = 20+1
name = os.path.join(current_dir, '../../data/credit_scoring/German/german.data')
train_size, dev_size, test_size = 0.7, 0.1, 0.2

if train_size + dev_size + test_size != 1:
    print("sample size wrong!!!")

# 每个数据的变量名
mean_list = ['Status of existing checking account', 'Duration in month', 'Credit history', 'Purpose',
             'Credit amount', 'Savings account/bonds', 'Present employment since',
             'Installment rate in percentage of disposable income', 'Personal status and sex',
             'Other debtors / guarantors', 'Present residence since', 'Property', 'Age in years',
             'Other installment plans', 'Housing', 'Number of existing credits at this bank' ,'Job',
             'Number of people being liable to provide maintenance for' , 'Telephone' , 'foreign worker',
             'target']

# data = pd.read_csv(name, sep=' ', names=[i for i in range(feature_size)])

# 原数据处理
# data中所有数据需要修改成数值格式
# todo age 和gender需要进一步划分成二分类？

# train_data = pd.read_csv('./bias_data/german_train.csv', sep=',', names=[i for i in range(feature_size)])
# train_data = predo(train_data)
# train = pd.DataFrame(train_data)
# train.columns = mean_list

# test_data = pd.read_csv('./bias_data/german_test.csv', sep=',', names=[i for i in range(feature_size)])
# test_data = predo(test_data)
# test = pd.DataFrame(test_data)
# # test = test.head(50)
# test.columns = mean_list # 表格重新写表头

# # method结果读取
# # todo 标签需要转换适配各个数据集
# # 'chatgpt/flare_german_desc/flare_german_desc_write_out_info.json'
# res = preres(test.values.tolist(), os.path.join(current_dir, './CALM/flare_german_desc_write_out_info.json'))
# res = pd.DataFrame(res)
# res.columns = mean_list
# print("Length of the result data:", len(res))

def prepare_input_data(filename):
    my_data = pd.read_csv(filename, sep=',', names=[i for i in range(feature_size)])
    my_data = predo(my_data)
    my_data_df = pd.DataFrame(my_data)
    my_data_df.columns = mean_list
    return my_data_df

def prepare_output_data(output_filename, my_test_df):
    my_data = preres(my_test_df.values.tolist(), output_filename)
    my_data_df = pd.DataFrame(my_data)
    my_data_df.columns = mean_list
    return my_data_df

train = prepare_input_data(os.path.join(current_dir, "bias_data", "german_train.csv"))
test = prepare_input_data(os.path.join(current_dir, "bias_data", "german_test.csv"))
res = prepare_output_data(os.path.join(current_dir, "CALM", "flare_german_desc_write_out_info.json"), test)

'''data bias test'''
# 测试数据本身偏见性
# favorable_label 为好的数值，即无风险的代表数字
# unfavorable_label 为坏的数值
# df 为数据
# label_names 作为目标的变量名
# protected_attribute_names 需要保护的变量名，含偏见的变量名
def disparate_impact(input_df, bias_attributes = ['Personal status and sex', 'Age in years','foreign worker']):
    input_data = BinaryLabelDataset(favorable_label=1, unfavorable_label=2, df=input_df, label_names=['target'], protected_attribute_names=bias_attributes)
    final_res = dict()
    
    # Foreigner DI
    metric = BinaryLabelDatasetMetric(input_data, unprivileged_groups=[{'foreign worker':0}], privileged_groups=[{'foreign worker':1}])
    # text_res = MetricTextExplainer(metric)        
    final_res['Foreigner'] = metric.disparate_impact()

    # Age DI
    metric = BinaryLabelDatasetMetric(input_data, unprivileged_groups=[{'Age in years':1}], privileged_groups=[{'Age in years':0}])
    # text_res = MetricTextExplainer(metric)        
    final_res['Age'] = metric.disparate_impact()

    # Gender DI
    metric = BinaryLabelDatasetMetric(input_data, unprivileged_groups=[{'Personal status and sex':1}], privileged_groups=[{'Personal status and sex':0}])
    # text_res = MetricTextExplainer(metric)        
    final_res['Gender'] = metric.disparate_impact()

    return final_res

print('data bias test for test data:')
print(disparate_impact(test))

# test_data = BinaryLabelDataset(favorable_label=1, unfavorable_label=2, df=test, label_names=['target'], protected_attribute_names=['Personal status and sex','Age in years','foreign worker'])

# # unprivileged_groups 弱势群体，例如{gender：1}表示弱势群体是女性，list[]内可以叠加，也可以多次使用分开算
# # privileged_groups 优势群体，例如{gender：2}表示优势群体是男性，
# metric = BinaryLabelDatasetMetric(test_data, unprivileged_groups=[{'foreign worker':0}], privileged_groups=[{'foreign worker':1}])
# text_res = MetricTextExplainer(metric)

# print('DI:', text_res.disparate_impact())

# metric = BinaryLabelDatasetMetric(test_data, unprivileged_groups=[{'Age in years':1}], privileged_groups=[{'Age in years':0}])
# text_res = MetricTextExplainer(metric)

# print('DI:', text_res.disparate_impact())

# metric = BinaryLabelDatasetMetric(test_data, unprivileged_groups=[{'Personal status and sex':1}], privileged_groups=[{'Personal status and sex':0}])
# text_res = MetricTextExplainer(metric)

# print('DI:', text_res.disparate_impact())

print('data bias test for train data:')
print(disparate_impact(train))

# train_data = BinaryLabelDataset(favorable_label=1, unfavorable_label=2, df=train, label_names=['target'], protected_attribute_names=['Personal status and sex','Age in years','foreign worker'])

# # unprivileged_groups 弱势群体，例如{gender：1}表示弱势群体是女性，list[]内可以叠加，也可以多次使用分开算
# # privileged_groups 优势群体，例如{gender：2}表示优势群体是男性，
# metric = BinaryLabelDatasetMetric(train_data, unprivileged_groups=[{'foreign worker':0}], privileged_groups=[{'foreign worker':1}])
# text_res = MetricTextExplainer(metric)

# print('DI:', text_res.disparate_impact())

# metric = BinaryLabelDatasetMetric(train_data, unprivileged_groups=[{'Age in years':1}], privileged_groups=[{'Age in years':0}])
# text_res = MetricTextExplainer(metric)

# print('DI:', text_res.disparate_impact())

# metric = BinaryLabelDatasetMetric(train_data, unprivileged_groups=[{'Personal status and sex':1}], privileged_groups=[{'Personal status and sex':0}])
# text_res = MetricTextExplainer(metric)

# print('DI:', text_res.disparate_impact())


'''method bias test'''
# 测试模型偏见性
# favorable_label 为好的数值，即无风险的代表数字
# unfavorable_label 为坏的数值
# df 为method输出的数据
# label_names 作为目标的变量名
# protected_attribute_names 需要保护的变量名，含偏见的变量名
def bias_test(output_df, input_test_df, bias_attributes = ['Personal status and sex', 'Age in years','foreign worker']):
    llm_output_data = BinaryLabelDataset(favorable_label=1, unfavorable_label=2, df=output_df, label_names=['target'], protected_attribute_names=bias_attributes)
    input_test_data = BinaryLabelDataset(favorable_label=1, unfavorable_label=2, df=input_test_df, label_names=['target'], protected_attribute_names=bias_attributes)
    final_res = {'EOD': {}, "AOD": {}}
    
    # Foreigner EOD and AOD
    metric = ClassificationMetric(input_test_data, llm_output_data, unprivileged_groups=[{'foreign worker':0}], privileged_groups=[{'foreign worker':1}])
    # text_res = MetricTextExplainer(metric)        
    final_res['EOD']["Foreigner"] = metric.equal_opportunity_difference()
    final_res['AOD']["Foreigner"] = metric.average_odds_difference()

    # Age EOD and AOD
    metric = ClassificationMetric(input_test_data, llm_output_data, unprivileged_groups=[{'Age in years':1}], privileged_groups=[{'Age in years':0}])
    # text_res = MetricTextExplainer(metric)      
    final_res['EOD']["Age"] = metric.equal_opportunity_difference()
    final_res['AOD']["Age"] = metric.average_odds_difference()

    # Gender EOD and AOD
    metric = ClassificationMetric(input_test_data, llm_output_data, unprivileged_groups=[{'Personal status and sex':1}], privileged_groups=[{'Personal status and sex':0}])
    # text_res = MetricTextExplainer(metric)        
    final_res['EOD']["Gender"] = metric.equal_opportunity_difference()
    final_res['AOD']["Gender"] = metric.average_odds_difference()

    return final_res

print('method bias test for test data:')
print(bias_test(res, test))

print("Results:", compute_metrics(os.path.join(current_dir, "CALM", "flare_german_desc_write_out_info.json")))
# res_data = BinaryLabelDataset(favorable_label=1, unfavorable_label=2, df=res, label_names=['target'], protected_attribute_names=['Personal status and sex','Age in years','foreign worker'])

# metric = ClassificationMetric(test_data, res_data, unprivileged_groups=[{'foreign worker':0}], privileged_groups=[{'foreign worker':1}])
# text_res = MetricTextExplainer(metric)

# print('EOD:', text_res.equal_opportunity_difference())
# print('ERR:', text_res.average_odds_difference())

# metric = ClassificationMetric(test_data, res_data, unprivileged_groups=[{'Age in years':1}], privileged_groups=[{'Age in years':0}])
# text_res = MetricTextExplainer(metric)

# print('EOD:', text_res.equal_opportunity_difference())
# print('ERR:', text_res.average_odds_difference())

# metric = ClassificationMetric(test_data, res_data, unprivileged_groups=[{'Personal status and sex':1}], privileged_groups=[{'Personal status and sex':0}])
# text_res = MetricTextExplainer(metric)

# print('EOD:', text_res.equal_opportunity_difference())
# print('ERR:', text_res.average_odds_difference())

print('down')