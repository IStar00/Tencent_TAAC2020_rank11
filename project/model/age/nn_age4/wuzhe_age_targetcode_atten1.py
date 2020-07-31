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
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
np.random.seed(1024)
rn.seed(1024)
tf.set_random_seed(1024)

#################################gpu22222222222222222222222#################

def get_age_data():
    train_user = pd.read_csv('train_preliminary/user.csv')
    train_final_user = pd.read_csv('train_semi_final/user.csv')
    test_data = test_data = pd.DataFrame({"user_id":(np.arange(3000001, 4000001, 1)), "age":"-1", "gender":"-1"})
    data = pd.concat([train_user,train_final_user, test_data], axis=0, sort=False)
    del train_user,train_final_user,test_data
    gc.collect()
    return data
data= get_age_data()
data = data.sort_values(by=['user_id']).reset_index(drop=True)

###############################################

emb1=np.load('256_w2c/w2v_256_creative_id_xuli95_win15iter8_emb.npy')
emb3=np.load('256_w2c/w2v_256_advertiser_id_xuli95_win15iter8.npy')
emb6=np.load('256_w2c/w2v_256_ad_id_xuli95_win15iter8_emb.npy')
emb7=np.load('256_w2c/naw2v_128_product_id_xuli95_win15iter8_emb.npy')##############128

x1=np.load('256_w2c/x1_xulie95_creative_id_win15iter8.npy')
x3=np.load('256_w2c/x5_xulie95_advertiser_id_win15iter8.npy')
x6=np.load('256_w2c/x3_xulie95_ad_id_win15iter8.npy')
x7=np.load('256_w2c/nax1_xulie95_product_id_win15iter8.npy') 
emb1=emb1.astype(np.float32)
emb3=emb3.astype(np.float32)
emb6=emb6.astype(np.float32)
emb7=emb7.astype(np.float32)
train_data = data[data['age']!='-1']
#train_data = data[data['gender']!='-1']

train_input_1 = x1[:len(train_data)]
#test_input_1 = x1[len(train_data):]
train_input_3 = x3[:len(train_data)]
#test_input_3 = x3[len(train_data):]
train_input_6 = x6[:len(train_data)]
#test_input_6 = x6[len(train_data):]

train_input_7 = x7[:len(train_data)]
#test_input_7 = x7[len(train_data):]

label = to_categorical(train_data['age'] - 1)
#label = to_categorical(train_data['gender'] - 1)

gc.collect()

#增强序列
x11=np.load('256_w2c/x1_xulie95_creative_id_reset_user_time_win15iter8.npy')
x33=np.load('256_w2c/x5_xulie95_advertiser_id_reset_user_time_win15iter8.npy')
x66=np.load('256_w2c/x3_xulie95_ad_id_reset_user_time_win15iter8.npy')
x77=np.load('256_w2c/nax1_xulie95_reset_userid_product_id_win15iter8.npy')
train_input_11 = x11[:len(train_data)]
#test_input_11 = x11[len(train_data):]

train_input_33 = x33[:len(train_data)]
#test_input_33 = x33[len(train_data):]

train_input_66 = x66[:len(train_data)]
#test_input_66 = x66[len(train_data):]
train_input_77 = x77[:len(train_data)]
#test_input_77 = x77[len(train_data):]
del x1,x3,x6,x11,x33,x66 , x7,x77
len_id=np.load('256_w2c/len_id.npy')
train_input_len = len_id[:len(train_data)]
#test_input_len = len_id[len(train_data):]
len_id_input=len_id.shape[1]
gc.collect()
#tfidf
import pickle
''''''
with open('tfidf/final_tfidf11_creative_id_age.pkl', 'rb') as f:
    f1 = pickle.load(f)  

with open('tfidf/final__tfidf11_ad_id_age.pkl', 'rb') as f:
    f2 = pickle.load(f)
with open('tfidf/final__tfidf11_advertiser_id_age.pkl', 'rb') as f:    
    f3 = pickle.load(f)

with open('tfidf/final_count11__ad_id_age.pkl', 'rb') as f:
    f4 = pickle.load(f)    
with open('tfidf/final_count11_advertiser_id_age.pkl', 'rb') as f:
    f5 = pickle.load(f)
with open('tfidf/final_count11_creative_id_age.pkl', 'rb') as f:    
    f6 = pickle.load(f) 
with open('256_w2c/final_target_4id_1age.pkl', 'rb') as f:
    f7 = pickle.load(f)
with open('256_w2c/final_target_5id_37age.pkl', 'rb') as f:    
    f8 = pickle.load(f) 

feature = pd.concat([f1,f2, f3,f4,f5,f6,f7,f8], axis=1, sort=False)
del f1,f2, f3,f4,f5,f6,f7,f8

gc.collect()
feature = feature.fillna(-1)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(feature)
hin_feature = ss.transform(feature)
num_feature_input = hin_feature.shape[1]

train_input_5 = hin_feature[:len(train_data)]
#test_input_5 = hin_feature[len(train_data):]
del feature,hin_feature
gc.collect()

###########
# 需要用到的函数
class AdamW(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4,  # decoupled weight decay (1/4)
                 epsilon=1e-8, decay=0., **kwargs):
        super(AdamW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            # decoupled weight decay (2/4)
            self.wd = K.variable(weight_decay, name='weight_decay')
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        wd = self.wd  # decoupled weight decay (3/4)

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            # decoupled weight decay (4/4)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) - lr * wd * p

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'weight_decay': float(K.get_value(self.wd)),
                  'epsilon': self.epsilon}
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


from keras.engine.topology import Layer
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

#########
from tensorflow.keras import backend as K
import numpy as np
#from keras.callbacks import ModelCheckpoint,TensorBoard
class CyclicLR(Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
    
from sklearn.model_selection import KFold
skf = KFold(n_splits=5, shuffle=True, random_state=2020)##kfold划分训练数据的索引
for i, (train_index, test_index) in enumerate(skf.split(train_input_5, train_data['age'].astype('int'))):
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
##主函数
def model_conv(emb1, emb3, emb6,emb7,num_feature_input,len_id_input): #emb3,  , num_feature_input
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
    emb_layer_7 = Embedding(
        input_dim=emb7.shape[0],
        output_dim=emb7.shape[1],
        weights=[emb7],
        input_length=95,
        trainable=False
    ) 
    seq1 = Input(shape=(95,))#####30
    seq3 = Input(shape=(95,))        
    seq6 = Input(shape=(95,))    
    seq7 = Input(shape=(95,))
    
    x1 = emb_layer_1(seq1)
    x3 = emb_layer_3(seq3)    
    x6 = emb_layer_6(seq6)
    x7 = emb_layer_7(seq7)
    

    sdrop=SpatialDropout1D(rate=0.2)
    x1 = sdrop(x1)
    x3 = sdrop(x3)
    x6 = sdrop(x6)
    x7 = sdrop(x7)
    
    x13 = concatenate([x1,x3,x6,x7], axis=-1 ) # lstm+attention结构 
    x13 = Dropout(0.35)(Bidirectional(CuDNNLSTM(300, return_sequences=True))(x13))       
    x13 = Dense(256, activation='relu')(x13)
    semantic13 = TimeDistributed(Dense(128, activation="tanh"))(x13)
    merged_13 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic13)
    len_mask=Input(shape=(len_id_input,))
    merged_13_avg = Lambda(lambda x: K.sum(x, axis=1)/len_mask, output_shape=(128,))(semantic13)
    x = Dropout(0.3)(Bidirectional(CuDNNLSTM(300, return_sequences=True))(x13))       
    att_1 = Attention(95)(x)
    att_1 = Dense(128, activation='relu')(att_1)

    hin = Input(shape=(num_feature_input, ))
    htime = Dense(256)(hin)
    x = concatenate([att_1, merged_13,  merged_13_avg, htime])
    x = Dropout(0.4)(Activation(activation="relu")(BatchNormalization()(Dense(1500)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(600)(x)))

    pred = Dense(10, activation='softmax')(x)######10
    model = Model(inputs=[seq1, seq3, seq6,seq7,hin,len_mask], outputs=pred)
    #from keras.utils import multi_gpu_model
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy',
                  optimizer=AdamW(lr=0.001,weight_decay=0.06,),metrics=["accuracy"])
    return model

gc.collect()
######一折
startload = time.time()
score = []  
count = 1     #不同折数  数字由从0到4                                    
train_index=num1_train ###########################10
test_index=num1_test ###########################10

print("FOLD | ", count+1)
print("###"*35)
gc.collect()
filepath = "age_model/nn_v2_lstm_tarcode_5id_atten1_%d.h5" % count
checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
reduce_lr = ReduceLROnPlateau(       
monitor='val_acc', factor=0.2, patience=2, min_lr=0.00001, verbose=1)##衰减 学习率

clr = CyclicLR(base_lr=0.000005, max_lr=0.002,
                        step_size=4687, mode='triangular2')#!!!!!!!!!!!!!!!!!!!!
earlystopping = EarlyStopping(
        monitor='val_acc', min_delta=0.0001, patience=4, verbose=1, mode='max')
callbacks = [checkpoint,reduce_lr, earlystopping]
 
model_age = model_conv(emb1, emb3,emb6,emb7, num_feature_input,len_id_input)############
    
model_age.summary()
x1_tr, x1_va = np.array(train_input_1)[train_index], np.array(train_input_1)[test_index] 
x3_tr, x3_va = np.array(train_input_3)[train_index], np.array(train_input_3)[test_index]#########
x6_tr, x6_va = np.array(train_input_6)[train_index], np.array(train_input_6)[test_index]#########
x7_tr, x7_va = np.array(train_input_7)[train_index], np.array(train_input_7)[test_index]#########
del  train_input_1,train_input_3,train_input_6,train_input_7
gc.collect()    
x11_tr, x11_va = np.array(train_input_11)[train_index], np.array(train_input_11)[test_index] 
x33_tr, x33_va = np.array(train_input_33)[train_index], np.array(train_input_33)[test_index]#########
x66_tr, x66_va = np.array(train_input_66)[train_index], np.array(train_input_66)[test_index]#########
x77_tr, x77_va = np.array(train_input_77)[train_index], np.array(train_input_77)[test_index]#########    
x5_tr, x5_va = np.array(train_input_5)[train_index], np.array(train_input_5)[test_index]
x9_len_tr, x9_len_va = np.array(train_input_len)[train_index], np.array(train_input_len)[test_index]
y_tr, y_va = label[train_index], label[test_index]

##减少内存
del  train_input_11,train_input_33,train_input_66,train_input_5,train_input_len,label,train_input_77
gc.collect()
da_x1_tr=np.vstack((x1_tr,x11_tr))
del  x1_tr, x11_tr
gc.collect()
da_x3_tr=np.vstack((x3_tr,x33_tr))
del  x3_tr, x33_tr
gc.collect()
da_x6_tr=np.vstack((x6_tr,x66_tr))
del  x6_tr,x66_tr
gc.collect()
da_x7_tr=np.vstack((x7_tr,x77_tr))
da_x7_va=np.vstack((x7_va,x77_va))
da_x1_va=np.vstack((x1_va,x11_va))
del  x1_va,x11_va , x7_tr,x77_tr,x7_va,x77_va
gc.collect()
da_x3_va=np.vstack((x3_va,x33_va))
del  x3_va,x33_va
gc.collect()
da_x6_va=np.vstack((x6_va,x66_va))
del  x6_va,x66_va
gc.collect()
da_x5_tr=np.vstack((x5_tr,x5_tr))
del  x5_tr
gc.collect()
da_x5_va=np.vstack((x5_va,x5_va))
da_x9_len_va=np.vstack((x9_len_va,x9_len_va))
da_x9_len_tr=np.vstack((x9_len_tr,x9_len_tr))
del  x5_va,x9_len_tr, x9_len_va
gc.collect()
da_y_tr=np.vstack((y_tr,y_tr))
del  y_tr
gc.collect()
da_y_va=np.vstack((y_va,y_va))
del  y_va
gc.collect()
hist = model_age.fit(  [da_x1_tr, da_x3_tr,da_x6_tr,da_x7_tr, da_x5_tr,da_x9_len_tr],     #
                         da_y_tr, batch_size=1024, epochs=50, 
                          validation_data=([da_x1_va, da_x3_va, da_x6_va,da_x7_va,da_x5_va,da_x9_len_va], da_y_va), #  validation_data=([x1_va, x5_va], y_va),
                         callbacks=callbacks, verbose=1, shuffle=True)
score.append(np.max(hist.history['val_acc']))
count += 1
print('acc:', np.mean(score))
endload = time.time() - startload
print('加载时间：%.2fmin' % (endload / 60))

#7.11