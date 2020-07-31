
#pip install gensim
#pip install keras==2.2.4
##1，功能：形构造每个用户的序列文件
import os
import pandas as pd
import gc
#############################！！train_preliminary！！#####################################
#1、读取数据 把所有数据合入到一起
train_user = pd.read_csv('train_preliminary/user.csv')
train_log = pd.read_csv('train_preliminary/click_log.csv')
train_ad = pd.read_csv('train_preliminary/ad.csv',na_values=['\\N'])
train_ad=train_ad.fillna(0)
############################！！test！！####################################
test_ad = pd.read_csv('test/ad.csv',na_values=['\\N'])
test_ad=test_ad.fillna(0)
test_log = pd.read_csv('test/click_log.csv')
##########################！！train_semi_final！！！#######################################
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
#合并训练和测试集数据为df
#############################！！！df！！!####################################
df=pd.concat([train_data, train_final_data,test_data], axis=0, sort=False)
del train_data, train_final_data,test_data
gc.collect()
#########
os.listdir("model/istar")
if not os.path.exists("model/istar/datapre"):
    os.mkdir("model/istar/datapre")
######################################################################
#2、把每个userid的序列聚合一起，构造两种不同顺序的序列
one=df
one = one.sort_values(by=['user_id'])##先按照 user_id来排序在join 形成的序列
def data_df_one(x):
    one[x]=one[x].astype(str)
    df_product_category=one.groupby(['user_id'])[x].apply(list).reset_index()
    df_product_category[x]=df_product_category[x].apply(lambda x:' '.join(x))
    df_product_category.to_csv("model/istar/datapre/df_reset_userid_{}.csv".format(x),index=None)
    print("已保存")
    gc.collect()
data_df_one('creative_id')
data_df_one('ad_id')
data_df_one('advertiser_id')

#无排序 .reset_index(drop=True)
df = df.sort_values(by=['user_id','time']).reset_index(drop=True) ####先按照 user_id、time 来排序在join形成的序列
def data_df(x):
    df[x]=df[x].astype(str)
    df_product_category=df.groupby(['user_id'])[x].apply(list).reset_index()
    df_product_category[x]=df_product_category[x].apply(lambda x:' '.join(x))
    df_product_category.to_csv("model/istar/datapre/df_reset_user_time_{}.csv".format(x),index=None)
    print("已保存")
    gc.collect()
data_df('creative_id')
data_df('ad_id')
data_df('advertiser_id')