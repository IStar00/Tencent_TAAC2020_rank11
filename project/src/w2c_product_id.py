'''
!pip install gensim
!pip install keras==2.2.4
!conda install -c conda-forge ipywidgets
路径可能需要调整下
描述1、对product id构造95的padding序列和训练128位的w2c ,和w2c_3id.py的差别仅在序列的顺序和w2c的size上
    2、生成一个padding后的序列中实际id的长度文件 len_id.npy
输入：原始数据集
输出：productid 的两种不同顺序的padding和w2c 已经每个padding中实际id的长度 len_id.npy
'''
import os
import pandas as pd
import numpy as np
import gc
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
##############################！！读取数据,合并数据集！！####################################
train_user = pd.read_csv('train_preliminary/user.csv')
train_log = pd.read_csv('train_preliminary/click_log.csv')
train_ad = pd.read_csv('train_preliminary/ad.csv')

test_ad = pd.read_csv('test/ad.csv')

test_log = pd.read_csv('test/click_log.csv')
train_final_user = pd.read_csv('train_semi_final/user.csv')
train_final_log = pd.read_csv('train_semi_final/click_log.csv')
train_final_ad = pd.read_csv('train_semi_final/ad.csv')#

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
##############################！！构造product序列！！####################################
os.listdir("model/istar")
if not os.path.exists("model/istar/datapre"):
    os.mkdir("model/istar/datapre")
tt=df    
tt = tt.sort_values(by=['user_id','time'])
def data_df(x):
    tt[x]=tt[x].astype(str)
    df_product_category=tt.groupby(['user_id'])[x].apply(list).reset_index()
    df_product_category[x]=df_product_category[x].apply(lambda x:' '.join(x))
    df_product_category.to_csv("model/istar/datapre/df_allid_xulie/df_noreset_index_na_{}.csv".format(x),index=None)
    print("已保存")
    gc.collect()
data_df('product_id')
############################################################################################################
##读取序列 去构造padding和w2c
def get_age_data():
    train_user = pd.read_csv('train_preliminary/user.csv')
    train_final_user = pd.read_csv('train_semi_final/user.csv')
    test_data = test_data = pd.DataFrame({"user_id":(np.arange(3000001, 4000001, 1)), "age":"-1", "gender":"-1"})
    data = pd.concat([train_user,train_final_user, test_data], axis=0, sort=False)
    del train_user,train_final_user,test_data
    gc.collect()
    return data
def get_df_product_id():    
    data = pd.read_csv('model/istar/datapre/df_noreset_index_na_product_id.csv')
    return data

id_label = get_age_data()
product_id = get_df_product_id()
data = pd.merge(id_label, product_id, on='user_id', how='left')

import os
import pandas as pd
from tqdm.autonotebook import *
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from gensim.models import FastText, Word2Vec
import re
from keras.layers import *
from keras.models import *
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import *
from keras.layers.advanced_activations import LeakyReLU, PReLU
import keras.backend as K
from keras.optimizers import *
from keras.utils import to_categorical
import tensorflow as tf
import random as rn
import gc
import logging
import gensim

### Tokenizer 序列化文本
def set_tokenizer(docs, split_char=' ', max_len=100):
    '''
    输入
    docs:文本列表
    split_char:按什么字符切割
    max_len:截取的最大长度
    
    输出
    X:序列化后的数据
    word_index:文本和数字对应的索引
    '''
    tokenizer = Tokenizer(lower=False, char_level=False, split=split_char)
    tokenizer.fit_on_texts(docs)
    X = tokenizer.texts_to_sequences(docs)
    maxlen = max_len
    X = pad_sequences(X, maxlen=maxlen, value=0)
    word_index=tokenizer.word_index
    return X, word_index

### 做embedding 这里采用word2vec 可以换成其他例如（glove词向量）
#from gensim.models.callbacks import CallbackAny2Vec
def trian_save_word2vec(docs, embed_size=128, save_name='w2v.txt', split_char=' '):
    '''
    输入
    docs:输入的文本列表
    embed_size:embed长度
    save_name:保存的word2vec位置
    
    输出
    w2v:返回的模型
    '''
    input_docs = []
    for i in docs:
        input_docs.append(i.split(split_char))
    logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    #w2v = Word2Vec(input_docs, size=embed_size, sg=1, window=15, seed=2020, workers=60, min_count=1,iter=7)
    w2v = Word2Vec(input_docs, size=embed_size, sg=1, window=15, seed=2020, workers=60, min_count=1,iter=8)
    
    
    w2v.wv.save_word2vec_format(save_name)
    print("w2v model done")
    return w2v

# 得到embedding矩阵
def get_embedding_matrix(word_index, embed_size=128, Emed_path="w2v_300.txt"):
    embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(
        Emed_path, binary=False)
    nb_words = len(word_index)+1
    embedding_matrix = np.zeros((nb_words, embed_size))
    count = 0
    for word, i in tqdm(word_index.items()):
        if i >= nb_words:
            continue
        try:
            embedding_vector = embeddings_index[word]
        except:
            embedding_vector = np.zeros(embed_size)
            count += 1
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector    
    print("null cnt",count)
    return embedding_matrix
############################################################################################################
#构造第一种序列 的padding和w2c
text_1_list = list(data['product_id'].astype(str))
print('开始序列化')
x1, index_1 = set_tokenizer(text_1_list, split_char=' ', max_len=95)
gc.collect()
np.save('model/istar/datapre/nax1_xulie95_product_id_win15iter8.npy', x1)
trian_save_word2vec(text_1_list, save_name='model/istar/datapre/naw2v_128_product_id_xuli95_win15iter8.txt', split_char=' ')
gc.collect()
emb1 = get_embedding_matrix(index_1, Emed_path='model/istar/datapre/naw2v_128_product_id_xuli95_win15iter8.txt')
np.save('model/istar/datapre/naw2v_128_product_id_xuli95_win15iter8_emb.npy', emb1)
print('emb完成')

############################################################################################################
#构造第二种序列 的padding
tt=df
tt = tt.sort_values(by=['user_id'])
def data_df(x):
    tt[x]=tt[x].astype(str)
    df_product_category=tt.groupby(['user_id'])[x].apply(list).reset_index()
    df_product_category[x]=df_product_category[x].apply(lambda x:' '.join(x))
    df_product_category.to_csv("model/istar/datapre/df_reset_userid_na_{}.csv".format(x),index=None)
    print("已保存")
    gc.collect()
data_df('product_id')

def get_df_product_id():    
    data = pd.read_csv('model/istar/datapre/df_allid_xulie/df_reset_userid_na_product_id.csv')
    return data
id_label = get_age_data()

product_id = get_df_product_id()
data = pd.merge(id_label, product_id, on='user_id', how='left')
text_1_list = list(data['product_id'].astype(str))
print('开始序列化')
x1, index_1 = set_tokenizer(text_1_list, split_char=' ', max_len=95)
gc.collect()
np.save('model/istar/datapre/nax1_xulie95_reset_userid_product_id_win15iter8.npy', x1)

############################################################################################################
#构造95序列中每个user——id的实际id的数量
x77=np.load('256_w2c/nax1_xulie95_reset_userid_product_id_win15iter8.npy')
memo = [[sum(i!=0)]  for i in x77 ]############len
np.array(memo)
np.save("'model/istar/dataprelen_id.npy", memo)