'''
构造w2c特征
!pip install gensim
!pip install keras==2.2.4
!conda install -c conda-forge ipywidgets
路径可能需要调整下
输入： df_data.py保存的两种不同顺序的文件
功能：只对**reset_user_time**文件训练256位的w2c 和emb，对***reset_userid*** 只形成padding文件
    序列 95 / w2v = Word2Vec(input_docs, size=embed_size, sg=1, window=15, seed=2020, workers=60, min_count=1,iter=8
输出 三种id（creative_id，advertiser_id，ad_id）的两种不同顺序的padding和三份256位的w2c
'''

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
##############################！！读取数据！！####################################
def get_age_data():##读取标签序列
    train_user = pd.read_csv('train_preliminary/user.csv')
    train_final_user = pd.read_csv('train_semi_final/user.csv')
    test_data = test_data = pd.DataFrame({"user_id":(np.arange(3000001, 4000001, 1)), "age":"-1", "gender":"-1"})
    data = pd.concat([train_user,train_final_user, test_data], axis=0, sort=False)
    del train_user,train_final_user,test_data
    gc.collect()
    return data

def get_df_ad_id():
    data = pd.read_csv('df_reset_user_time_ad_id.csv')
    return data
def get_df_advertiser_id():    
    data = pd.read_csv('df_reset_user_time_advertiser_id.csv')
    return data
def get_df_creative_id():   
    data = pd.read_csv('df_reset_user_time_creative_id.csv')
    return data

id_label = get_age_data()
ad_id = get_df_ad_id()
advertiser_id = get_df_advertiser_id()
creative_id = get_df_creative_id()

data = pd.merge(id_label, creative_id, on='user_id', how='left')
data = pd.merge(data, advertiser_id, on='user_id', how='left')
data = pd.merge(data, ad_id, on='user_id', how='left')

##########################！！！padding,w2c,emb函数！！！############################################
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
def trian_save_word2vec(docs, embed_size=256, save_name='w2v.txt', split_char=' '):
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
    w2v = Word2Vec(input_docs, size=embed_size, sg=1, window=15, seed=2020, workers=60, min_count=1,iter=8)
    
    w2v.wv.save_word2vec_format(save_name)
    print("w2v model done")
    return w2v

# 得到embedding矩阵
def get_embedding_matrix(word_index, embed_size=256, Emed_path="w2v_300.txt"):
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
###############################！！！保存文件结果！！！#######################################
### 
'''
本次形成的序列95 ，w2c size为256
w2v = Word2Vec(input_docs, size=embed_size, sg=1, window=15, seed=2020, workers=60, min_count=1,iter=8
'''
text_1_list = list(data['creative_id'].astype(str))
text_3_list = list(data['ad_id'].astype(str))
text_5_list = list(data['advertiser_id'].astype(str)) 
#text_6_list = list(data['product_id'].astype(str))


print('开始序列化')
x1, index_1 = set_tokenizer(text_1_list, split_char=' ', max_len=95)###x1为 padding后序列文件 index为对应字典
x3, index_3 = set_tokenizer(text_3_list, split_char=' ', max_len=95)
x5, index_5 = set_tokenizer(text_5_list, split_char=' ', max_len=95)
#x6, index_6 = set_tokenizer(text_6_list, split_char=' ', max_len=95)
gc.collect()
np.save('w2c/x1_xulie95_creative_id_win15iter8.npy', x1)
np.save('w2c/x3_xulie95_ad_id_win15iter8.npy', x3)
np.save('w2c/x5_xulie95_advertiser_id_win15iter8.npy', x5)

### w2c 二进制文件
trian_save_word2vec(text_1_list, save_name='w2c/w2v_256_creative_id_xuli95_win15iter8.txt', split_char=' ')
trian_save_word2vec(text_3_list, save_name='w2c/w2v_256_ad_id_xuli95_win15iter8.txt', split_char=' ')
trian_save_word2vec(text_5_list, save_name='w2c/w2v_256_advertiser_id_xuli95_win15iter8.txt', split_char=' ')

emb1 = get_embedding_matrix(index_1, Emed_path='w2c/w2v_256_creative_id_xuli95_win15iter8.txt')
emb3 = get_embedding_matrix(index_3, Emed_path='w2c/w2v_256_ad_id_xuli95_win15iter8.txt')
emb5 = get_embedding_matrix(index_5, Emed_path='w2c/w2v_256_advertiser_id_xuli95_win15iter8.txt')
### emb文件
gc.collect()
np.save('w2c/w2v_256_creative_id_xuli95_win15iter8_emb.npy', emb1)
np.save('w2c/w2v_256_ad_id_xuli95_win15iter8_emb.npy', emb3)
np.save('w2c/w2v_256_advertiser_id_xuli95_win15iter8.npy', emb5)

print('emb完成')

##########对第二种序列仅形成padding文件即可 不需要训练w2c
def get_age_data():
    train_user = pd.read_csv('train_preliminary/user.csv')
    train_final_user = pd.read_csv('train_semi_final/user.csv')
    test_data = test_data = pd.DataFrame({"user_id":(np.arange(3000001, 4000001, 1)), "age":"-1", "gender":"-1"})
    data = pd.concat([train_user,train_final_user, test_data], axis=0, sort=False)
    del train_user,train_final_user,test_data
    gc.collect()
    return data
def get_df_ad_id():
    data = pd.read_csv('df_reset_userid_ad_id.csv')
    return data
def get_df_advertiser_id():    
    data = pd.read_csv('df_reset_userid_advertiser_id.csv')
    return data
def get_df_creative_id():   
    data = pd.read_csv('df_reset_userid_creative_id.csv')
###############################合并序列文件#######################################
id_label = get_age_data()
ad_id = get_df_ad_id()
advertiser_id = get_df_advertiser_id()
creative_id = get_df_creative_id()
data = pd.merge(id_label, creative_id, on='user_id', how='left')
data = pd.merge(data, advertiser_id, on='user_id', how='left')
data = pd.merge(data, ad_id, on='user_id', how='left')
###############################保存序列化文件#######################################
text_1_list = list(data['creative_id'].astype(str))
text_3_list = list(data['ad_id'].astype(str))
text_5_list = list(data['advertiser_id'].astype(str)) 
print('开始序列化')
x1, index_1 = set_tokenizer(text_1_list, split_char=' ', max_len=95)
x3, index_3 = set_tokenizer(text_3_list, split_char=' ', max_len=95)
x5, index_5 = set_tokenizer(text_5_list, split_char=' ', max_len=95)
gc.collect()
np.save('w2c/x1_xulie95_creative_id_reset_user_time_win15iter8.npy', x1)
np.save('w2c/x3_xulie95_ad_id_reset_user_time_win15iter8.npy', x3)
np.save('w2c/x5_xulie95_advertiser_id_reset_user_time_win15iter8.npy', x5)