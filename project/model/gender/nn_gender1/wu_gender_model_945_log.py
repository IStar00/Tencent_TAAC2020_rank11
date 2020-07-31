!pip install gensim
!pip install keras==2.2.4
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
#import tensorflow.compat.v1 as tf #2.1
#tf.disable_v2_behavior()#2.1

import random as rn
import gc
import logging
import gensim

#from tensorflow.contrib.rnn import *
os.environ['PYTHONHASHSEED'] = '0'
# 显卡使用（如没显卡需要注释掉）
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
np.random.seed(1024)
rn.seed(1024)
tf.set_random_seed(1024)
#tf.random.set_seed(1024)##2.1
#path="data/"
#os.listdir("data/")
#################################gpu22222222222222222222222#################

def get_age_data():
    train_user = pd.read_csv('train_preliminary/user.csv')
    train_final_user = pd.read_csv('train_semi_final/user.csv')
    test_data =  pd.DataFrame({"user_id":(np.arange(3000001, 4000001, 1)), "age":"-1", "gender":"-1"})
    data = pd.concat([train_user,train_final_user, test_data], axis=0, sort=False)
    del train_user,train_final_user,test_data
    gc.collect()
    return data
data= get_age_data()
data = data.sort_values(by=['user_id']).reset_index(drop=True)
###############################################
emb1=np.load('w2c/w2v_256_creative_id_xuli95_win15iter8_emb.npy')
emb3=np.load('w2c/w2v_256_advertiser_id_xuli95_win15iter8.npy')
emb6=np.load('w2c/w2v_256_ad_id_xuli95_win15iter8_emb.npy')

x1=np.load('w2c/x1_xulie95_creative_id_win15iter8.npy')
x3=np.load('w2c/x5_xulie95_advertiser_id_win15iter8.npy')
x6=np.load('w2c/x3_xulie95_ad_id_win15iter8.npy')
 
emb1=emb1.astype(np.float32)
emb3=emb3.astype(np.float32)
emb6=emb6.astype(np.float32)

#train_data = data[data['age']!='-1']
train_data = data[data['gender']!='-1']

train_input_1 = x1[:len(train_data)]
test_input_1 = x1[len(train_data):]
train_input_3 = x3[:len(train_data)]
test_input_3 = x3[len(train_data):]
train_input_6 = x6[:len(train_data)]
test_input_6 = x6[len(train_data):]
#train_input_5 = hin_feature[:len(train_data)]
#test_input_5 = hin_feature[len(train_data):]

#label = to_categorical(train_data['age'] - 1)
label = to_categorical(train_data['gender'] - 1)

gc.collect()

#增强序列
x11=np.load('w2c/x1_xulie95_creative_id_reset_user_time_win15iter8.npy')
x33=np.load('w2c/x5_xulie95_advertiser_id_reset_user_time_win15iter8.npy')
x66=np.load('w2c/x3_xulie95_ad_id_reset_user_time_win15iter8.npy')

train_input_11 = x11[:len(train_data)]
test_input_11 = x11[len(train_data):]

train_input_33 = x33[:len(train_data)]
test_input_33 = x33[len(train_data):]

train_input_66 = x66[:len(train_data)]
test_input_66 = x66[len(train_data):]

del x1,x3,x6,x11,x33,x66
gc.collect()

#tfidf
import pickle
with open('tfidf/final_tfidf11_creative_id_gender.pkl', 'rb') as f:
    f1 = pickle.load(f)    
with open('tfidf/final__tfidf11_ad_id_gender.pkl', 'rb') as f:
    f2 = pickle.load(f)
with open('tfidf/final__tfidf11_advertiser_id_gender.pkl', 'rb') as f:    
    f3 = pickle.load(f)
    
feature = pd.concat([f1,f2, f3], axis=1, sort=False)

feature = feature.fillna(-1)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(feature)
hin_feature = ss.transform(feature)
num_feature_input = hin_feature.shape[1]

train_input_5 = hin_feature[:len(train_data)]
test_input_5 = hin_feature[len(train_data):]
del f1,f2, f3,feature
gc.collect()

skf = StratifiedKFold(n_splits=5, random_state=1010, shuffle=True)
for i, (train_index, test_index) in enumerate(skf.split(train_input_5, train_data['gender'].astype('int'))):
    if(i==0):
        num0_train=train_index
        num0_test=test_index
    elif(i==1):
        num1_train=train_index
        num1_test=test_index
    elif(i==2):
        num2_train=train_index
        num2_test=test_index
    elif(i==3):
        num3_train=train_index
        num3_test=test_index
    elif(i==4):
        num4_train=train_index
        num4_test=test_index
    del train_index, test_index
gc.collect()

def model_conv(emb1, emb3, emb6, num_feature_input): #emb3,  , num_feature_input
    K.clear_session()
    emb_layer_1 = Embedding(
        input_dim=emb1.shape[0],
        output_dim=emb1.shape[1],
        weights=[emb1],
        input_length=95,###30
        trainable=False
    )   
    emb_layer_3 = Embedding(
        input_dim=emb3.shape[0],
        output_dim=emb3.shape[1],
        weights=[emb3],
        input_length=95,
        trainable=False
    )    
    emb_layer_6 = Embedding(
        input_dim=emb6.shape[0],
        output_dim=emb6.shape[1],
        weights=[emb6],
        input_length=95,
        trainable=False
    ) 
    seq1 = Input(shape=(95,))#####30
    seq3 = Input(shape=(95,))        
    seq6 = Input(shape=(95,))    
    x1 = emb_layer_1(seq1)
    x3 = emb_layer_3(seq3)    
    x6 = emb_layer_6(seq6)
    sdrop=SpatialDropout1D(rate=0.2)
    x1 = sdrop(x1)
    x3 = sdrop(x3)
    x6 = sdrop(x6)
    x11 = concatenate([x1,x3,x6] , axis=-1)           
    x = Dropout(0.4)(Bidirectional(CuDNNLSTM(300, return_sequences=True))(x11))       
    x = Dense(256, activation='relu')(x)
    semantic = TimeDistributed(Dense(100, activation="tanh"))(x)
    
    merged_1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(100,))(semantic)
    merged_1_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(100,))(semantic)
    hin = Input(shape=(num_feature_input, ))
    #htime = Dense(16, activation='relu')(hin)
    htime = Dense(128)(hin)
    #x = concatenate([merged_1,  merged_1_avg])   
    x = concatenate([merged_1,  merged_1_avg, htime])
    
    x = Dropout(0.4)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))

    #pred = Dense(10, activation='softmax')(x)######10
    pred = Dense(2, activation='softmax')(x)
    model = Model(inputs=[seq1, seq3, seq6,hin], outputs=pred)
    #model = Model(inputs=[seq1, seq3, seq6], outputs=pred)
    #model = Model(inputs=[seq1, hin], outputs=pred)
    from keras.utils import multi_gpu_model
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy',
                  optimizer=AdamW(lr=0.001,weight_decay=0.06,),metrics=["accuracy"])
    return model
gc.collect()
model_age = model_conv(emb1, emb3,emb6,num_feature_input)

def dataloader(test_index):

    x1_va = np.array(train_input_1)[test_index] 

    x3_va = np.array(train_input_3)[test_index]#########
    x6_va = np.array(train_input_6)[test_index]#########

    x11_va =np.array(train_input_11)[test_index] 
    x33_va =np.array(train_input_33)[test_index]#########
    x66_va =np.array(train_input_66)[test_index]#########
    
    x5_va = np.array(train_input_5)[test_index]
    y_va =label[test_index]
    gc.collect()
    return x1_va, x3_va,x6_va, x5_va,x11_va, x33_va,x66_va,y_va

sub = np.zeros((test_input_1.shape[0], 2))###########################2
oof_pred = np.zeros((train_input_1.shape[0], 2))###########################2
oof_pred_da = np.zeros((train_input_1.shape[0], 2))###########################10
sub_da = np.zeros((test_input_1.shape[0], 2)) ###########################10

startload = time.time()

filepath = "gender_model/nn_v1_0.h5" 
model_age.load_weights(filepath)
x1_va, x3_va,x6_va, x5_va,x11_va, x33_va,x66_va ,y_va=dataloader(num0_test)

oof_pred[num0_test] = model_age.predict([x1_va, x3_va,x6_va, x5_va],batch_size=1024,verbose=1)###2048
sub += model_age.predict([test_input_1, test_input_3,test_input_6,test_input_5],batch_size=1024,verbose=1)/skf.n_splits##   
oof_pred_da[num0_test] = model_age.predict([x11_va, x33_va,x66_va, x5_va],batch_size=1024,verbose=1)
sub_da += model_age.predict([test_input_11, test_input_33,test_input_66,test_input_5],batch_size=1024,verbose=1)/skf.n_splits
    
filepath = "gender_model/nn_v1_1.h5" 
model_age.load_weights(filepath)
x1_va, x3_va,x6_va, x5_va,x11_va, x33_va,x66_va ,y_va=dataloader(num1_test)

oof_pred[num1_test] = model_age.predict([x1_va, x3_va,x6_va, x5_va],batch_size=1024,verbose=1)###2048
sub += model_age.predict([test_input_1, test_input_3,test_input_6,test_input_5],batch_size=1024,verbose=1)/skf.n_splits##
oof_pred_da[num1_test] = model_age.predict([x11_va, x33_va,x66_va, x5_va],batch_size=1024,verbose=1)
sub_da += model_age.predict([test_input_11, test_input_33,test_input_66,test_input_5],batch_size=1024,verbose=1)/skf.n_splits
###########################


filepath = "gender_model/nn_v1_2.h5" 
model_age.load_weights(filepath)
x1_va, x3_va,x6_va, x5_va,x11_va, x33_va,x66_va ,y_va=dataloader(num2_test)

oof_pred[num2_test] = model_age.predict([x1_va, x3_va,x6_va, x5_va],batch_size=1024,verbose=1)###2048
sub += model_age.predict([test_input_1, test_input_3,test_input_6,test_input_5],batch_size=1024,verbose=1)/skf.n_splits##
oof_pred_da[num2_test] = model_age.predict([x11_va, x33_va,x66_va, x5_va],batch_size=1024,verbose=1)
sub_da += model_age.predict([test_input_11, test_input_33,test_input_66,test_input_5],batch_size=1024,verbose=1)/skf.n_splits
 ###########################   
    
filepath = "gender_model/nn_v1_3.h5" 
model_age.load_weights(filepath)
x1_va, x3_va,x6_va, x5_va,x11_va, x33_va,x66_va ,y_va=dataloader(num3_test)

oof_pred[num3_test] = model_age.predict([x1_va, x3_va,x6_va, x5_va],batch_size=1024,verbose=1)###2048
sub += model_age.predict([test_input_1, test_input_3,test_input_6,test_input_5],batch_size=1024,verbose=1)/skf.n_splits##
oof_pred_da[num3_test] = model_age.predict([x11_va, x33_va,x66_va, x5_va],batch_size=1024,verbose=1)
sub_da += model_age.predict([test_input_11, test_input_33,test_input_66,test_input_5],batch_size=1024,verbose=1)/skf.n_splits
###########################

filepath = "gender_model/nn_v1_4.h5" 
model_age.load_weights(filepath)
x1_va, x3_va,x6_va, x5_va,x11_va, x33_va,x66_va ,y_va=dataloader(num4_test)

oof_pred[num4_test] = model_age.predict([x1_va, x3_va,x6_va, x5_va],batch_size=1024,verbose=1)###2048
sub += model_age.predict([test_input_1, test_input_3,test_input_6,test_input_5],batch_size=1024,verbose=1)/skf.n_splits##
oof_pred_da[num4_test] = model_age.predict([x11_va, x33_va,x66_va, x5_va],batch_size=1024,verbose=1)
sub_da += model_age.predict([test_input_11, test_input_33,test_input_66,test_input_5],batch_size=1024,verbose=1)/skf.n_splits
    
endload = time.time() - startload
print('加载时间：%.2fmin' % (endload / 60))

accuracy_score(np.argmax((label),axis=1)+1, np.argmax((oof_pred),axis=1)+1)
accuracy_score(np.argmax((label),axis=1)+1, np.argmax((oof_pred_da),axis=1)+1)
score=accuracy_score(np.argmax((label),axis=1)+1, np.argmax((oof_pred_da+oof_pred),axis=1)+1)
aa=sub+sub_da
test = data[data['gender'] =='-1']
submit = test[['user_id']]
submit.columns = ['user_id']
submit['predicted_gender'] = aa.argmax(1)+1

#五折
oof = np.concatenate((oof_pred,sub))
oof = pd.DataFrame(oof)
oof.columns = [str(i+1) for i in range(2)]
oof['id'] = pd.concat([train_data[['user_id']],test[['user_id']]])['user_id'].values
#oof.to_csv("sub/wuzhe_submission_nn_gender3_1input_fc256_drop0.4_bn0.4_datatime_relr_{}.csv".format(np.mean(score)),index=False) #v1_test.csv
np.save("sub/wuzhe_submission_nn_gender3_1input_fc256_drop0.4_bn0.4_datatime_relr_{}.npy".format(np.mean(score)), oof)

#五折
oof1 = np.concatenate((oof_pred_da,sub_da))
oof1 = pd.DataFrame(oof1)
oof1.columns = [str(i+1) for i in range(2)]
oof1['id'] = pd.concat([train_data[['user_id']],test[['user_id']]])['user_id'].values
#oof1.to_csv("sub/wuzhe_submission_nn_gender3_1input_fc256_drop0.4_bn0.4_datatime_relr_dastr_{}.csv".format(np.mean(score)),index=False) #v1_test.csv
np.save("sub/wuzhe_submission_nn_gender3_1input_fc256_drop0.4_bn0.4_datatime_relr_dastr_{}.npy".format(np.mean(score)), oof1)