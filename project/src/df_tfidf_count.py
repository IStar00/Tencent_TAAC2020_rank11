
##功能：统计userid中3id的count 和tfidf特征 
'''
pip install tqdm
对每个useid的序列当作一篇文章 统计每种类别id的count和tfidf特征
输入： 函数get_user_creative_data(x): 即df_data.py 形成的三个id的聚合文件
过程： 先统计每个userid的tfidf 和count 然后通过七个弱模型stacking形成每个id的age有70列特征 ，gender有14列特征
模型（'LogisticRegression','SGDClassifier'，'PassiveAggressiveClassifier'，'RidgeClassfiy'，'LinearSVC'，'BernoulliNB'，'MultinomialNB)
输出：对3个id(creative_id，advertiser_id，ad_id)的tfidf和count特征 保存为pkl格式 压缩空间
'''
import json
import pandas as pd
import os
from tqdm import *
import re
import numpy as np
import gc
import warnings
import pickle
warnings.filterwarnings("ignore")
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
######################### 模型函数(返回sklean_stacking结果) ########################
def get_age_data():
    train_user = pd.read_csv('train_preliminary/user.csv')
    train_final_user = pd.read_csv('train_semi_final/user.csv')
    test_data = test_data = pd.DataFrame({"user_id":(np.arange(3000001, 4000001, 1)), "age":"-1", "gender":"-1"})
    data = pd.concat([train_user,train_final_user, test_data], axis=0, sort=False)
    del train_user,train_final_user,test_data
    gc.collect()
    return data
def get_user_creative_data(x):
    data = pd.read_csv('model/istar/datapre/df_reset_userid_'+x+'.csv')
    return data
user_data = get_age_data()

######################### 模型函数(返回sklean_stacking结果) ########################
def get_sklearn_classfiy_stacking(clf, train_feature, test_feature, score, model_name, class_number, n_folds, train_num, test_num):
    print('\n****开始跑', model_name, '****')
    stack_train = np.zeros((train_num, class_number))
    stack_test = np.zeros((test_num, class_number))
    score_mean = []
    skf = StratifiedKFold(n_splits=n_folds, random_state=1017)
    tqdm.desc = model_name
    for i, (tr, va) in enumerate(skf.split(train_feature, score)):
        clf.fit(train_feature[tr], score[tr])
        if(model_name=='BernoulliNB'): 
            score_va = clf.predict_proba(train_feature[va])
            score_te = clf.predict_proba(test_feature)
            score_single = accuracy_score(score[va], np.argmax(clf.predict_proba(train_feature[va]), axis=1))
        elif(model_name=='MultinomialNB'):
            score_va = clf.predict_proba(train_feature[va])
            score_te = clf.predict_proba(test_feature)
            score_single = accuracy_score(score[va], np.argmax(clf.predict_proba(train_feature[va]), axis=1))
        else:
            score_va = clf._predict_proba_lr(train_feature[va])
            score_te = clf._predict_proba_lr(test_feature)
            score_single = accuracy_score(score[va], np.argmax(clf._predict_proba_lr(train_feature[va]), axis=1))
        score_mean.append(np.around(score_single, 5))
        print(f'{i+1}/{n_folds}', score_single)
        stack_train[va] += score_va
        stack_test += score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])
    df_stack = pd.DataFrame()
    for i in range(stack.shape[1]):
        df_stack['tfidf_ori_1_1_' + model_name + '_classfiy_{}'.format(i)] = stack[:, i]
    print(model_name, '处理完毕')
    gc.collect()
    return df_stack, score_mean

model_list = [
    ['LogisticRegression', LogisticRegression(random_state=1017, C=3)],
    ['SGDClassifier', SGDClassifier(random_state=1017, loss='log')],
    ['PassiveAggressiveClassifier', PassiveAggressiveClassifier(random_state=1017, C=2)],
    ['RidgeClassfiy', RidgeClassifier(random_state=1017)],
    ['LinearSVC', LinearSVC(random_state=1017)],
    ['BernoulliNB', BernoulliNB()],
    ['MultinomialNB', MultinomialNB()]    
    
###############################对gender构造tfidf特征 #########################################
def tfidf_data_gender(df_id):
    tqdm.pandas('获取特征')
    # userid序列表计算tfidf作为特征
    data = get_user_creative_data(df_id)
    data = pd.merge(user_data, data, on='user_id', how='left')
    df_train = data[data['gender']!= '-1']
    df_test = data[data['gender']=='-1']
    ############################ 加载数据 ############################
    data = pd.concat([df_train, df_test], axis=0, sort=False)
    data[df_id] = data[df_id].apply(lambda row:str(row))
    ############################ tf-idf ############################
    print('开始计算tf-idf特征')
    tfidf_vec = TfidfVectorizer(ngram_range=(1,1))
    discuss_tf = tfidf_vec.fit_transform(data[df_id]).tocsr()
    del tfidf_vec
    gc.collect()
    print('计算结束')
    print('开始进行一些前期处理')
    train_feature = discuss_tf[:len(df_train)]
    score = df_train['gender']-1
    test_feature = discuss_tf[len(df_train):]
    print('处理完毕')
    feature = pd.DataFrame()
    for i in model_list:
        stack_result, score_mean = get_sklearn_classfiy_stacking(i[1], train_feature, test_feature, score.astype('int'), i[0], 2, 5, len(df_train), len(df_test))
        feature = pd.concat([feature, stack_result], axis=1, sort=False)
        print('五折结果', score_mean)
        print('平均结果', np.mean(score_mean))    
    return feature
#########
os.listdir("model")
if not os.path.exists("model/tfidf"):
    os.mkdir("model/tfidf")
#######保存结果############################
feature=tfidf_data_gender('ad_id')    
test1 = r'model/tfidf/final__tfidf11_ad_id_gender.pkl' 
with open(test1,'wb') as f:  
        pickle.dump(feature,f,protocol = 4) 

feature=tfidf_data_gender('creative_id')    
test1 = r'model/tfidf/final_tfidf11_creative_id_gender.pkl'  
with open(test1,'wb') as f:  
        pickle.dump(feature,f,protocol = 4) 
    
feature=tfidf_data_gender('advertiser_id')    
test1 = r'model/tfidf/final__tfidf11_advertiser_id_gender.pkl'   
with open(test1,'wb') as f:  
        pickle.dump(feature,f,protocol = 4) 

########################对age构造tfidf特征##########################################
def tfidf_data_age(df_id):
    tqdm.pandas('获取特征')
    # userid序列表计算tfidf作为特征
    data = get_user_creative_data(df_id)
    data = pd.merge(user_data, data, on='user_id', how='left')
    df_train = data[data['age']!= '-1']
    df_test = data[data['age']=='-1']
    ############################ 加载数据 ############################
    data = pd.concat([df_train, df_test], axis=0, sort=False)
    data[df_id] = data[df_id].apply(lambda row:str(row))
    ############################ tf-idf ############################
    print('开始计算tf-idf特征')
    tfidf_vec = TfidfVectorizer(ngram_range=(1,1))
    discuss_tf = tfidf_vec.fit_transform(data[df_id]).tocsr()
    del tfidf_vec
    gc.collect()
    print('计算结束')
    print('开始进行一些前期处理')
    train_feature = discuss_tf[:len(df_train)]
    score = df_train['age']-1
    test_feature = discuss_tf[len(df_train):]
    print('处理完毕')
    feature = pd.DataFrame()
    for i in model_list:
        stack_result, score_mean = get_sklearn_classfiy_stacking(i[1], train_feature, test_feature, score.astype('int'), i[0], 10, 5, len(df_train), len(df_test))
        feature = pd.concat([feature, stack_result], axis=1, sort=False)
        print('五折结果', score_mean)
        print('平均结果', np.mean(score_mean))
    return feature
#######保存结果############################
feature=tfidf_data_age('ad_id')    
test1 = r'model/tfidf/final__tfidf11_ad_id_age.pkl' 
with open(test1,'wb') as f:  
        pickle.dump(feature,f,protocol = 4) 

feature=tfidf_data_age('creative_id')    
test1 = r'model/tfidf/final_tfidf11_creative_id_age.pkl'  
with open(test1,'wb') as f:  
        pickle.dump(feature,f,protocol = 4) 
    
feature=tfidf_data_age('advertiser_id')    
test1 = r'model/tfidf/final__tfidf11_advertiser_id_age.pkl'  
with open(test1,'wb') as f:  
        pickle.dump(feature,f,protocol = 4) 
    

########################对age构造count特征##########################################
def count_data_age(df_id):
    tqdm.pandas('获取特征')
    # userid序列表计算tfidf作为特征
    data = get_user_creative_data(df_id)
    data = pd.merge(user_data, data, on='user_id', how='left')
    df_train = data[data['age']!= '-1']
    df_test = data[data['age']=='-1']
    ############################ 加载数据 ############################
    data = pd.concat([df_train, df_test], axis=0, sort=False)
    data[df_id] = data[df_id].apply(lambda row:str(row))
    ############################ tf-idf ############################
    print('开始计算count特征')
    count_vec = CountVectorizer(ngram_range=(1,1))
    discuss_tf = count_vec.fit_transform(data[df_id]).tocsr()
    del tfidf_vec
    gc.collect()
    print('计算结束')
    print('开始进行一些前期处理')
    train_feature = discuss_tf[:len(df_train)]
    score = df_train['age']-1
    test_feature = discuss_tf[len(df_train):]
    print('处理完毕')
    feature = pd.DataFrame()
    for i in model_list:
        stack_result, score_mean = get_sklearn_classfiy_stacking(i[1], train_feature, test_feature, score.astype('int'), i[0], 10, 5, len(df_train), len(df_test))
        feature = pd.concat([feature, stack_result], axis=1, sort=False)
        print('五折结果', score_mean)
        print('平均结果', np.mean(score_mean))
    return feature
#######保存结果############################
feature=count_data_age('ad_id')    
test1 = r'model/tfidf/final_count11__ad_id_age.pkl' 
with open(test1,'wb') as f:  
        pickle.dump(feature,f,protocol = 4) 

feature=count_data_age('creative_id')    
test1 = r'model/tfidf/final_count11_creative_id_age.pkl'  
with open(test1,'wb') as f:  
        pickle.dump(feature,f,protocol = 4) 
    
feature=count_data_age('advertiser_id')    
test1 = r'model/tfidf/final_count11_advertiser_id_age.pkl'  
with open(test1,'wb') as f:  
        pickle.dump(feature,f,protocol = 4) 

###############################对gender构造count特征 #########################################
def count_data_gender(df_id):
    tqdm.pandas('获取特征')
    # userid序列表计算tfidf作为特征
    data = get_user_creative_data(df_id)
    data = pd.merge(user_data, data, on='user_id', how='left')
    df_train = data[data['gender']!= '-1']
    df_test = data[data['gender']=='-1']
    ############################ 加载数据 ############################
    data = pd.concat([df_train, df_test], axis=0, sort=False)
    data[df_id] = data[df_id].apply(lambda row:str(row))
    ############################ tf-idf ############################
    print('开始计算count特征')
    count_vec = CountVectorizer(ngram_range=(1,1))
    discuss_tf = count_vec.fit_transform(data[df_id]).tocsr()
    del tfidf_vec
    gc.collect()
    print('计算结束')
    print('开始进行一些前期处理')
    train_feature = discuss_tf[:len(df_train)]
    score = df_train['gender']-1
    test_feature = discuss_tf[len(df_train):]
    print('处理完毕')
    feature = pd.DataFrame()
    for i in model_list:
        stack_result, score_mean = get_sklearn_classfiy_stacking(i[1], train_feature, test_feature, score.astype('int'), i[0], 2, 5, len(df_train), len(df_test))
        feature = pd.concat([feature, stack_result], axis=1, sort=False)
        print('五折结果', score_mean)
        print('平均结果', np.mean(score_mean))    
    return feature
#########
os.listdir("model")
if not os.path.exists("model/tfidf"):
    os.mkdir("model/tfidf")
#######保存结果############################
feature=count_data_gender('ad_id')    
test1 = r'model/tfidf/final_count11__ad_id_gender.pkl' 
with open(test1,'wb') as f:  
        pickle.dump(feature,f,protocol = 4) 

feature=count_data_gender('creative_id')    
test1 = r'model/tfidf/final_count11_creative_id_gender.pkl'  
with open(test1,'wb') as f:  
        pickle.dump(feature,f,protocol = 4) 
    
feature=count_data_gender('advertiser_id')    
test1 = r'model/tfidf/final_count11_advertiser_id_gender.pkl'   
with open(test1,'wb') as f:  
        pickle.dump(feature,f,protocol = 4) 
