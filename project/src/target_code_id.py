'''
描述：对age gender进行目标编码特征提取
pandas版本0.24.2（小于0.25版本）
pip install tqdm
输入：原始数据集
过程：五折提取不同id对应age gender标签的['mean', 'std', 'mad','median','max','min','skew','count']等特征，最后进行均值处理。
输出：gender的目标编码一个（adid）  age目标编码文件三个，（crea和adid各一个 其他advertiser_id', 'product_id', 'industry','product_category'合起来作为一个文件）

'''
import warnings
warnings.filterwarnings("ignore")
from tqdm import *
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
#import tqdm
from sklearn.metrics import roc_auc_score
import scipy
import gc

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
##############################！！读取数据！！####################################
train_user = pd.read_csv('train_preliminary/user.csv')
train_log = pd.read_csv('train_preliminary/click_log.csv')
train_ad = pd.read_csv('train_preliminary/ad.csv',na_values=['\\N'])
train_ad=train_ad.fillna(0)
test_ad = pd.read_csv('test/ad.csv',na_values=['\\N'])
test_ad=test_ad.fillna(0)
test_log = pd.read_csv('test/click_log.csv')

train_final_user = pd.read_csv('train_semi_final/user.csv')
train_final_log = pd.read_csv('train_semi_final/click_log.csv')
train_final_ad = pd.read_csv('train_semi_final/ad.csv',na_values=['\\N'])#
train_final_ad=train_final_ad.fillna(0)

data = pd.merge(train_log, train_ad, on='creative_id', how='left')
train_data = pd.merge(data, train_user, on='user_id', how='left')
del train_log,train_ad,data,train_user
gc.collect()
data_final = pd.merge(train_final_log, train_final_ad, on='creative_id', how='left')
train_final_data = pd.merge(data_final, train_final_user, on='user_id', how='left')
del train_final_log,train_final_ad,data_final,train_final_user
gc.collect()

test_data = pd.merge(test_log, test_ad, on='creative_id', how='left')
test_data['age']=-1
test_data['gender']=-1
df=pd.concat([train_data, train_final_data,test_data], axis=0, sort=False)
df = df.sort_values(by=['user_id','time'])

###########################################11111111111##################################################################
##############################！！针对adid进行gender的目标编码！！####################################
########################################################################################################################
from sklearn.model_selection import KFold
train_df = df[df['gender']!=-1].reset_index(drop=True)
test_df = df[df['gender']==-1].reset_index(drop=True)

def group_feature(df, key, target, aggs):  ##对不同id进行聚合提取特征 
    agg_dict = {}
    for ag in aggs:
        agg_dict[f'{target}_{ag}'] = ag
    print(agg_dict)
    t = df.groupby(key)[target].agg(agg_dict).reset_index()
    return t

### 对gender的ad——id 提取['mean', 'std', 'mad','median','max','min','skew','count']的特征
enc_cols = []
stats_default_dict = {
    'max': train_df['gender'].max(),
    'min': train_df['gender'].min(),
    'median': train_df['gender'].median(),
    'mean': train_df['gender'].mean(),
    'sum': train_df['gender'].sum(),
    'std': train_df['gender'].std(),
    'skew': train_df['gender'].skew(),
    'kurt': train_df['gender'].kurt(),
    'count': train_df['gender'].count(),    
    'mad': train_df['gender'].mad()
}
##########################################################
##########划分5折进行提取特征
enc_stats = ['mean', 'std', 'mad','median','max','min','skew','count']
skf = KFold(n_splits=5, shuffle=True, random_state=2020)   ##/ssd/wa.pkl 
for f in tqdm(['ad_id']): #######
    enc_dict = {}
    for stat in enc_stats:
        enc_dict['{}_target_{}'.format(f, stat)] = stat
        train_df['{}_target_{}'.format(f, stat)] = 0
        test_df['{}_target_{}'.format(f, stat)] = 0
        enc_cols.append('{}_target_{}'.format(f, stat))
    for i, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['gender'])):
        trn_x, val_x = train_df.iloc[trn_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)
        enc_df = trn_x.groupby(f, as_index=False)['gender'].agg(enc_dict)
        val_x = val_x[[f]].merge(enc_df, on=f, how='left')
        test_x = test_df[[f]].merge(enc_df, on=f, how='left')
        for stat in enc_stats:
            val_x['{}_target_{}'.format(f, stat)] = val_x['{}_target_{}'.format(f, stat)].fillna(stats_default_dict[stat])
            test_x['{}_target_{}'.format(f, stat)] = test_x['{}_target_{}'.format(f, stat)].fillna(stats_default_dict[stat])
            train_df.loc[val_idx, '{}_target_{}'.format(f, stat)] = val_x['{}_target_{}'.format(f, stat)].values
            test_df['{}_target_{}'.format(f, stat)] += test_x['{}_target_{}'.format(f, stat)].values / skf.n_splits
print(train_df.shape,'train_df.shape')
############################################################
###构造每个用户的特征表格
col=['user_id','time','age','gender']
train_id=df[col]
train_id.drop_duplicates(['user_id'], keep='first', inplace=True)
train_id_df=train_id[:3000000]
data_col=['creative_id', 'click_times','ad_id', 'product_id','product_category','advertiser_id','industry']
num_cat_col=[i  for i in train_df.columns  if i not in col ]
tar_col_qa=[i  for i in num_cat_col  if i not in data_col ]
print('特征')
print(tar_col_qa)
########################################################
#分别对train 和test提取特征
for i in tar_col_qa:
    taa = group_feature(train_df, 'user_id',i,['mean'])
    train_id_df = pd.merge(train_id_df, taa, on='user_id', how='left') 
#test 
test_id_df=train_id[3000000:]
for i in tar_col_qa:
    taa = group_feature(test_df, 'user_id',i,['mean'])
    test_id_df = pd.merge(test_id_df, taa, on='user_id', how='left') 
    
###统计每个user 90天点击广告的id count特征
taa = group_feature(train_df, 'user_id','ad_id_target_mean',['count'])
train_id_df = pd.merge(train_id_df, taa, on='user_id', how='left') 

taa = group_feature(test_df, 'user_id','ad_id_target_mean',['count'])
test_id_df = pd.merge(test_id_df, taa, on='user_id', how='left') 
#########################################################
###合并特征数据 保存pkl格式
col=['user_id','time','age','gender']
numcol=[i  for i in test_id_df.columns  if i not in col ]
tar_id=train_id_df[numcol]
tar_id_test=test_id_df[numcol]
feat = pd.concat([tar_id,tar_id_test], axis=0, sort=False,ignore_index=True)
os.listdir("model")
if not os.path.exists("model/tfidf"):
    os.mkdir("model/tfidf")
import pickle
test1 = r'model/tfidf/final_target_ad_id_9_gender.pkl'   
with open(test1,'wb') as f:  
    pickle.dump(feat,f,protocol = 4)
    
    
#######################################22222222222###################################################################    
##############################！！针对adid进行age的目标编码！！####################################
######################################################################################################################
from sklearn.model_selection import KFold
train_df = df[df['age']!=-1].reset_index(drop=True)
test_df = df[df['age']==-1].reset_index(drop=True)

### 对age的ad——id 提取['mean', 'std', 'mad','median','max','min','skew','count']的特征
enc_cols = []
stats_default_dict = {
    'max': train_df['age'].max(),
    'min': train_df['age'].min(),
    'median': train_df['age'].median(),
    'mean': train_df['age'].mean(),
    'sum': train_df['age'].sum(),
    'std': train_df['age'].std(),
    'skew': train_df['age'].skew(),
    'kurt': train_df['age'].kurt(),
    'count': train_df['age'].count(),    
    'mad': train_df['age'].mad()
}
##############################################
##########划分5折进行提取特征
##############################################
enc_stats = ['mean', 'std', 'mad','median','max','min','skew','count']
skf = KFold(n_splits=5, shuffle=True, random_state=2020)   ##/final_target_5id_37age.pkl 
for f in tqdm(['ad_id']):#######

    enc_dict = {}
    for stat in enc_stats:
        enc_dict['{}_target_{}'.format(f, stat)] = stat
        train_df['{}_target_{}'.format(f, stat)] = 0
        test_df['{}_target_{}'.format(f, stat)] = 0
        enc_cols.append('{}_target_{}'.format(f, stat))
    for i, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['age'])):
        trn_x, val_x = train_df.iloc[trn_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)
        enc_df = trn_x.groupby(f, as_index=False)['age'].agg(enc_dict)
        val_x = val_x[[f]].merge(enc_df, on=f, how='left')
        test_x = test_df[[f]].merge(enc_df, on=f, how='left')
        for stat in enc_stats:
            val_x['{}_target_{}'.format(f, stat)] = val_x['{}_target_{}'.format(f, stat)].fillna(stats_default_dict[stat])
            test_x['{}_target_{}'.format(f, stat)] = test_x['{}_target_{}'.format(f, stat)].fillna(stats_default_dict[stat])
            train_df.loc[val_idx, '{}_target_{}'.format(f, stat)] = val_x['{}_target_{}'.format(f, stat)].values
print(train_df.shape,'train_df.shape')
######################################################
###构造每个用户的特征表格
col=['user_id','time','age','gender']
train_id=df[col]
train_id.drop_duplicates(['user_id'], keep='first', inplace=True)
train_id_df=train_id[:3000000]
data_col=['creative_id', 'click_times','ad_id', 'product_id','product_category','advertiser_id','industry']
num_cat_col=[i  for i in train_df.columns  if i not in col ]
tar_col_qa=[i  for i in num_cat_col  if i not in data_col ]
print('特征')
print(tar_col_qa)
###############################################
#分别对train 和test提取特征
for i in tar_col_qa:
    taa = group_feature(train_df, 'user_id',i,['mean'])
    train_id_df = pd.merge(train_id_df, taa, on='user_id', how='left') 
#test 
test_id_df=train_id[3000000:]
for i in tar_col_qa:
    taa = group_feature(test_df, 'user_id',i,['mean'])
    test_id_df = pd.merge(test_id_df, taa, on='user_id', how='left') 
    
###统计每个user 90天点击广告的id count特征
taa = group_feature(train_df, 'user_id','ad_id_target_mean',['count'])
train_id_df = pd.merge(train_id_df, taa, on='user_id', how='left') 

taa = group_feature(test_df, 'user_id','ad_id_target_mean',['count'])
test_id_df = pd.merge(test_id_df, taa, on='user_id', how='left') 
################################################
###合并特征数据 保存pkl格式
col=['user_id','time','age','gender']
numcol=[i  for i in test_id_df.columns  if i not in col ]
tar_id=train_id_df[numcol]
tar_id_test=test_id_df[numcol]
feat = pd.concat([tar_id,tar_id_test], axis=0, sort=False,ignore_index=True)
test1 = r'model/tfidf/final_target_5id_37age.pkl'   
with open(test1,'wb') as f:  
    pickle.dump(feat,f,protocol = 4) 
    
#######################################333333333333###################################################################    
##############################！！针对creative——id进行age的目标编码！！####################################
########################################################################################################################
from sklearn.model_selection import KFold
train_df = df[df['age']!=-1].reset_index(drop=True)
test_df = df[df['age']==-1].reset_index(drop=True)

### 对age的ad——id 提取['mean', 'std', 'mad','median','max','min','skew','count']的特征
enc_cols = []
stats_default_dict = {
    'max': train_df['age'].max(),
    'min': train_df['age'].min(),
    'median': train_df['age'].median(),
    'mean': train_df['age'].mean(),
    'sum': train_df['age'].sum(),
    'std': train_df['age'].std(),
    'skew': train_df['age'].skew(),
    'kurt': train_df['age'].kurt(),
    'count': train_df['age'].count(),    
    'mad': train_df['age'].mad()
}
#################################################
##########划分5折进行提取特征
enc_stats = ['mean', 'std', 'mad','median','max','min','skew']
skf = KFold(n_splits=5, shuffle=True, random_state=2020)
for f in tqdm(['creative_id']):### 保存为final_target_crea_id_8age.pkl

    enc_dict = {}
    for stat in enc_stats:
        enc_dict['{}_target_{}'.format(f, stat)] = stat
        train_df['{}_target_{}'.format(f, stat)] = 0
        test_df['{}_target_{}'.format(f, stat)] = 0
        enc_cols.append('{}_target_{}'.format(f, stat))
    for i, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['age'])):
        trn_x, val_x = train_df.iloc[trn_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)
        enc_df = trn_x.groupby(f, as_index=False)['age'].agg(enc_dict)
        val_x = val_x[[f]].merge(enc_df, on=f, how='left')
        test_x = test_df[[f]].merge(enc_df, on=f, how='left')
        for stat in enc_stats:
            val_x['{}_target_{}'.format(f, stat)] = val_x['{}_target_{}'.format(f, stat)].fillna(stats_default_dict[stat])
            test_x['{}_target_{}'.format(f, stat)] = test_x['{}_target_{}'.format(f, stat)].fillna(stats_default_dict[stat])
            train_df.loc[val_idx, '{}_target_{}'.format(f, stat)] = val_x['{}_target_{}'.format(f, stat)].values
print(train_df.shape,'train_df.shape')
###################################################
###构造每个用户的特征表格
col=['user_id','time','age','gender']
train_id=df[col]
train_id.drop_duplicates(['user_id'], keep='first', inplace=True)
train_id_df=train_id[:3000000]
data_col=['creative_id', 'click_times','ad_id', 'product_id','product_category','advertiser_id','industry']
num_cat_col=[i  for i in train_df.columns  if i not in col ]
tar_col_qa=[i  for i in num_cat_col  if i not in data_col ]
print('特征')
print(tar_col_qa)
##################################################
#分别对train 和test提取特征
for i in tar_col_qa:
    taa = group_feature(train_df, 'user_id',i,['mean'])
    train_id_df = pd.merge(train_id_df, taa, on='user_id', how='left') 
#test 
test_id_df=train_id[3000000:]
for i in tar_col_qa:
    taa = group_feature(test_df, 'user_id',i,['mean'])
    test_id_df = pd.merge(test_id_df, taa, on='user_id', how='left') 
    
###统计每个user 90天点击广告的id count特征
taa = group_feature(train_df, 'user_id','creative_id_target_mean',['count'])
train_id_df = pd.merge(train_id_df, taa, on='user_id', how='left')

taa = group_feature(test_df, 'user_id','creative_id_target_mean',['count'])
test_id_df = pd.merge(test_id_df, taa, on='user_id', how='left') 
###############################################
###合并特征数据 保存pkl格式
col=['user_id','time','age','gender']
numcol=[i  for i in test_id_df.columns  if i not in col ]
tar_id=train_id_df[numcol]
tar_id_test=test_id_df[numcol]
feat = pd.concat([tar_id,tar_id_test], axis=0, sort=False,ignore_index=True)
import pickle
test1 = r'model/tfidf/final_target_crea_id_8age.pkl'   
with open(test1,'wb') as f:  
    pickle.dump(feat,f,protocol = 4)
    

#######################################！！444444444444444444444！！##################################################################################    
##############################！！针对['advertiser_id', 'product_id', 'industry','product_category']进行age的目标编码！！####################################
########################################################################################################################
from sklearn.model_selection import KFold
train_df = df[df['age']!=-1].reset_index(drop=True)
test_df = df[df['age']==-1].reset_index(drop=True)

### 对age的'advertiser_id', 'product_id', 'industry','product_category' 提取['mean', 'std', 'mad','median','max','min','skew','count']的特征
enc_cols = []
stats_default_dict = {
    'max': train_df['age'].max(),
    'min': train_df['age'].min(),
    'median': train_df['age'].median(),
    'mean': train_df['age'].mean(),
    'sum': train_df['age'].sum(),
    'std': train_df['age'].std(),
    'skew': train_df['age'].skew(),
    'kurt': train_df['age'].kurt(),
    'count': train_df['age'].count(),    
    'mad': train_df['age'].mad()
}
#################################################
##########划分5折进行提取特征
enc_stats = ['mean', 'std', 'mad','median','max','min','skew']
skf = KFold(n_splits=5, shuffle=True, random_state=2020)
for f in tqdm(['advertiser_id', 'product_id', 'industry','product_category']):### 保存为final_target_4id_1age.pkl

    enc_dict = {}
    for stat in enc_stats:
        enc_dict['{}_target_{}'.format(f, stat)] = stat
        train_df['{}_target_{}'.format(f, stat)] = 0
        test_df['{}_target_{}'.format(f, stat)] = 0
        enc_cols.append('{}_target_{}'.format(f, stat))
    for i, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['age'])):
        trn_x, val_x = train_df.iloc[trn_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)
        enc_df = trn_x.groupby(f, as_index=False)['age'].agg(enc_dict)
        val_x = val_x[[f]].merge(enc_df, on=f, how='left')
        test_x = test_df[[f]].merge(enc_df, on=f, how='left')
        for stat in enc_stats:
            val_x['{}_target_{}'.format(f, stat)] = val_x['{}_target_{}'.format(f, stat)].fillna(stats_default_dict[stat])
            test_x['{}_target_{}'.format(f, stat)] = test_x['{}_target_{}'.format(f, stat)].fillna(stats_default_dict[stat])
            train_df.loc[val_idx, '{}_target_{}'.format(f, stat)] = val_x['{}_target_{}'.format(f, stat)].values
print(train_df.shape,'train_df.shape')
###################################################
###构造每个用户的特征表格
col=['user_id','time','age','gender']
train_id=df[col]
train_id.drop_duplicates(['user_id'], keep='first', inplace=True)
train_id_df=train_id[:3000000]
data_col=['creative_id', 'click_times','ad_id', 'product_id','product_category','advertiser_id','industry']
num_cat_col=[i  for i in train_df.columns  if i not in col ]
tar_col_qa=[i  for i in num_cat_col  if i not in data_col ]
print('特征')
print(tar_col_qa)
##################################################
#分别对train 和test提取特征
for i in tar_col_qa:
    taa = group_feature(train_df, 'user_id',i,['mean'])
    train_id_df = pd.merge(train_id_df, taa, on='user_id', how='left') 
#test 
test_id_df=train_id[3000000:]
for i in tar_col_qa:
    taa = group_feature(test_df, 'user_id',i,['mean'])
    test_id_df = pd.merge(test_id_df, taa, on='user_id', how='left') 
    
###############################################
###合并特征数据 保存pkl格式
col=['user_id','time','age','gender']
numcol=[i  for i in test_id_df.columns  if i not in col ]
tar_id=train_id_df[numcol]
tar_id_test=test_id_df[numcol]
feat = pd.concat([tar_id,tar_id_test], axis=0, sort=False,ignore_index=True)
import pickle
test1 = r'model/tfidf/final_target_4id_1age.pkl'   
with open(test1,'wb') as f:  
    pickle.dump(feat,f,protocol = 4)