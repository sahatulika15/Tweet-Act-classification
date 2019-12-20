#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from transformers import BertModel, BertTokenizer, WordpieceTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


# In[ ]:


import pandas as pd
import numpy as np
import os
import pickle
import numpy as np
import re
import itertools
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# In[ ]:


os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
device='cuda:1'
device = torch.device(device if torch.cuda.is_available() else "cpu")
print('Hey:: device:', device)

# In[ ]:


pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)


# In[ ]:


model = BertModel.from_pretrained(pretrained_weights)
model=model.to(device)


# In[ ]:


train_path_DAC = './Data_bert/SWBD.csv'
dataframe=pd.read_csv(train_path_DAC)
X_train_swbd=dataframe['TEXT'].fillna('<UNK>').values
Y_train_swbd=dataframe['NEW_TAX'].fillna('<UNK>').values

test_path_DAC='./Data_bert/SWBD_valid.csv'
dataframe=pd.read_csv(test_path_DAC)
X_test_swbd=dataframe['TEXT'].fillna('<UNK>').values
Y_test_swbd=dataframe['NEW_TAX'].fillna('<UNK>').values

train_path_tw = './Data_bert/TWEET_DATA1.csv'
dataframe=pd.read_csv(train_path_tw)
X_train_tw=dataframe['TEXT'].fillna('<UNK>').values
Y_train_tw=dataframe['Label'].fillna('<UNK>').values


# In[ ]:


X_train_swbd_inp_ids = [  tokenizer.encode(sent, add_special_tokens=True, max_length=128) for sent in X_train_swbd  ]
X_test_swbd_inp_ids = [  tokenizer.encode(sent, add_special_tokens=True, max_length=128) for sent in X_test_swbd  ]
X_train_tw_inp_ids = [  tokenizer.encode(sent.lower(), add_special_tokens=True, max_length=128) for sent in X_train_tw  ]


# In[ ]:


maxLen = 128


# In[ ]:


# X_train_swbd_inp_ids_padded = pad_sequences(X_train_swbd_inp_ids, maxlen=maxLen, dtype="long", truncating="post", padding="post")
# X_test_swbd_inp_ids_padded = pad_sequences(X_test_swbd_inp_ids, maxlen=maxLen, dtype="long", truncating="post", padding="post")
X_train_tw_inp_ids_padded = pad_sequences(X_train_tw_inp_ids, maxlen=maxLen, dtype="long", truncating="post", padding="post")


# In[ ]:


def retAttnMask(input_ids):
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
      seq_mask = [float(i>0) for i in seq]
      attention_masks.append(seq_mask)
    return attention_masks


# In[ ]:


# X_train_swbd_inp_ids_attMask = retAttnMask(X_train_swbd_inp_ids_padded)
# X_test_swbd_inp_ids_attMask = retAttnMask(X_test_swbd_inp_ids_padded)
X_train_tw_inp_ids_attMask = retAttnMask(X_train_tw_inp_ids_padded)


# In[ ]:


def getData(X, M):
    x_t = torch.tensor(X)
    m_t = torch.tensor(M)
    mydata = TensorDataset(x_t, m_t)
    data_sampler = SequentialSampler(mydata)
    data_loader = DataLoader(mydata, sampler=data_sampler, batch_size=(512+128))
    return data_loader


# In[ ]:


# swbd_tr_dl = getData(X_train_swbd_inp_ids_padded, X_train_swbd_inp_ids_attMask)


# In[ ]:


# swbd_te_dl = getData(X_test_swbd_inp_ids_padded, X_test_swbd_inp_ids_attMask)

#swbd_te_dl = getData(X_test_swbd_inp_ids_padded, X_test_swbd_inp_ids_attMask)

tw_tr_dl = getData(X_train_tw_inp_ids_padded, X_train_tw_inp_ids_attMask)
# In[ ]:


model.eval()


# In[ ]:


def getEmbeddings(dl):
    sqbd_tr_emb = []
    cnt = 0
    for batch in dl:
        cnt+=1
        if cnt%1 == 0:
            print(cnt, end=" ", flush=True)
        
        with torch.no_grad():
            b_input_ids, b_input_mask = batch
            b_input_ids, b_input_mask = b_input_ids.to(device), b_input_mask.to(device)
            myout, _ = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            myout = myout.to(torch.device("cpu"))
            sqbd_tr_emb.append(myout.numpy())
    return sqbd_tr_emb


# In[ ]:


def get_split_inds(path):
    with open(path,'r') as f:
        a = f.readlines()
    a = a[1:]
    cnt = 0
    sp_ind = [0]
    for lin in a:
        if lin == '\n':
            sp_ind.append(cnt)
        else:
            cnt+=1
    return sp_ind


def split_data_dialog(values,unpaded_values,labels,sp_inds):
    li = []
    yli = []
    uli = []
    dial_len = []
    sentLen = []
    for i in range(len(sp_inds)-1):
        li.append(values[sp_inds[i]:sp_inds[i+1]])
        yli.append(labels[sp_inds[i]:sp_inds[i+1]])
        uli.append(unpaded_values[sp_inds[i]:sp_inds[i+1]])
        
        dial_len.append(len(li[-1]))
        st = []
        for i in unpaded_values[sp_inds[i]:sp_inds[i+1]]:
            st.append(len(i))
        sentLen.append(np.asarray(st,dtype = np.int32))
            
    return li,yli,uli,dial_len,sentLen


# In[ ]:


def create_label(label,lbl_Dict = None):
    
    if lbl_Dict is None:
        lbl_dict={}
        index=0
        for dial_lbls in label:
            if dial_lbls not in lbl_dict:
                lbl_dict[dial_lbls]=index
                index=index+1
    else:
        lbl_dict = lbl_Dict
    print(lbl_dict)
    Y=[]
    for i in label:
        xxx=np.zeros(int(len(lbl_dict)))
        j=lbl_dict.get(i)
        xxx[j]=1
        Y.append(xxx)
    return Y,lbl_dict
    
    
def split_data_dialog_tw(values,unpaded_values,labels,sp_inds, exf):
    li = []
    yli = []
    uli = []
    dial_len = []
    sentLen = []
    exfList = []
    li2 = []
    values2 = np.arange(len(values))
    for i in range(len(sp_inds)-1):
        li2.append(values2[sp_inds[i]:sp_inds[i+1]])
        li.append(values[sp_inds[i]:sp_inds[i+1]])
        exfList.append(exf[sp_inds[i]:sp_inds[i+1]])
        yli.append(labels[sp_inds[i]:sp_inds[i+1]])
        uli.append(unpaded_values[sp_inds[i]:sp_inds[i+1]])
        
        dial_len.append(len(li[-1]))
        st = []
        for j in unpaded_values[sp_inds[i]:sp_inds[i+1]]:
            st.append(len(j))
        sentLen.append(np.asarray(st,dtype = np.int32))
            
    return li,(li2,yli,uli,dial_len,sentLen,exfList)


# In[ ]:


# Y_train_swbd,ldic_swbd = create_label(Y_train_swbd,None)
# Y_test_swbd,_ = create_label(Y_test_swbd,ldic_swbd)
Y_train_tw,ldic_tw = create_label(Y_train_tw,None) 


# In[ ]:


ind_tr_swbd = get_split_inds(train_path_DAC)
ind_te_swbd = get_split_inds(test_path_DAC)


# In[ ]:


def save_each_dial(X_list, savepath):
    li  = X_list[0]
    lab = X_list[1]
    
    for i,dial in enumerate(li):
        np.save(savepath +'/X_'+str(i) +'.npy', np.asarray(dial, dtype = np.float32))
        np.save(savepath +'/Y_'+str(i) +'.npy', np.asarray(lab[i], dtype = np.int32))
        


# In[ ]:


def get_save_embeddings(dl, X_unpadded, ylab, sp_ind,exf, savepath):
    b = getEmbeddings(dl)
    c = np.concatenate(b,axis=0)
    X_data, X_List = split_data_dialog_tw(c,X_unpadded,ylab,sp_ind, exf)
    save_each_dial([X_data, X_List[1]], savepath)
    X_List = list(X_List)
    X_List.append(range(len(X_List[0])) )
    print(len(X_List))
    tw_tr_val = train_test_split(*X_List,test_size=0.2,random_state=42)
    to_save_tw_test = [
        tw_tr_val[1],
        tw_tr_val[3],
        tw_tr_val[9],
        tw_tr_val[7],
        ldic_tw,
        tw_tr_val[11],
        tw_tr_val[13],
    ]
    
    to_save_tw_train = [
        tw_tr_val[0],
        tw_tr_val[2],
        tw_tr_val[8],
        tw_tr_val[6],
        ldic_tw,
        tw_tr_val[10],
        tw_tr_val[12],
    ]
    with open('./Data_bert/Data/twit_validset_bert.pkl','wb') as f:
        pickle.dump(to_save_tw_test,f)
    with open('Data_bert/Data/twit_trainset_bert.pkl','wb') as f:
        pickle.dump(to_save_tw_train,f)
    
    


# In[ ]:
print('gonna do train set now!!')
num = 7500//32
ind_tr_tw = [ (i+1)*32 for i in range(num)]
ind_tr_tw = [0] +  ind_tr_tw
ind_tr_tw.append(7500)
#get_save_embeddings(swbd_tr_dl, X_train_swbd_inp_ids, Y_train_swbd, ind_tr_swbd, './Data_bert/swbd_tr')


# In[ ]:




with open('./Data_bert/final_data.pkl','rb') as g:
    extafeats = pickle.load(g)

extrafeats = [np.asarray(b.split(" "),dtype = np.float32) for b in extafeats]
extrafeats = np.asarray(extrafeats,dtype = np.float32)

get_save_embeddings(tw_tr_dl, X_train_tw_inp_ids, Y_train_tw, ind_tr_tw, extrafeats,'./Data_bert/tw_tr')


# In[ ]:




