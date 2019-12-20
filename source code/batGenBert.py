import numpy as np
import sys
from sklearn.utils import shuffle
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder
import math



class BatchGenerator():

    def __init__(self,path,embPath,batchSize,taskid,totTaskNum,isDial = False,hasExtra = False,doShuffle = False):

        '''

        :param path:
        :param batchSize:
        :param taskid:
        :param totTaskNum:
        :param isDial:
        :param doShuffle:
        '''

        self.hasExtra = hasExtra
        self.isDial = isDial
        self.path = path
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.taskid = taskid
        self.embPath = embPath
        doShuffle = False #//////////
        with open(self.path,'rb') as fd:
            allinfo = pkl.load(fd,encoding='bytes')
        fd.close() 
        
        self.curbatch = 0
        self.curdial = 0
        self.dialIds = allinfo[-1]
        self.x = np.load(self.embPath+ '/X_'+str(self.dialIds[self.curdial]) +'.npy')#= allinfo[0] # no of samples X num of utterances X words X embeddings
        self.y = allinfo[1] # no of samples X no of utterances X no oc classes
        
        self.sentlen = allinfo[2] # nof of samples X no of utterances
        self.dialLen = allinfo[3] # no of samples
        self.extrafeats = None
        if self.hasExtra :
            self.extrafeats = allinfo[5]
        del allinfo
        if doShuffle:
            if hasExtra:
                self.x, self.y, self.sentlen, self.dialLen , self.extrafeats = shuffle(self.x, self.y, self.sentlen, self.dialLen,self.extrafeats)
            else:
                self.x,self.y,self.sentlen,self.dialLen = shuffle(self.x,self.y,self.sentlen,self.dialLen)

        self.nSamp = len(self.y)
        if 1==1: # self.isDial
            cnt = 0
            for dial in self.dialLen:
                cnt+=math.ceil(dial/self.batchSize)
            self.numBatches = cnt
        else:
            self.numBatches = self.nSamp // self.batchSize
        ohe = onehot = OneHotEncoder(categories=[range(totTaskNum)], dtype=np.float32, sparse=False)
        self.ohe = onehot
        self.taskDiscTar = onehot.fit_transform(np.zeros([self.batchSize,1],dtype=np.int32)+taskid)




    def nextBatch(self):

        curextra = None

        if 1==1:
            beg = self.curbatch*self.batchSize
            fin = min(len(self.x),(self.curbatch+1) * self.batchSize)
            curx = self.x[beg:fin]
            cury = self.y[self.curdial][beg:fin]
            cursent = np.asarray(self.sentlen[self.curdial][beg:fin],dtype = np.int32)

            curdial = np.asarray([len(curx)],dtype = np.int32)
            taskdis = self.ohe.fit_transform(np.zeros([len(curx), 1], dtype=np.int32) + self.taskid)
            if self.isDial:
                curx = np.asarray([curx],dtype = np.float32)
                cury = np.asarray([cury],dtype = np.float32)
                cursent = np.asarray([cursent],dtype = np.int32)
            else:
                curx = np.asarray(curx,dtype = np.float32)
                cury = np.asarray(cury,dtype = np.float32)
                cursent = np.asarray(cursent,dtype = np.int32)

            
            if self.extrafeats is not None:
                curextra = self.extrafeats[self.curdial][beg:fin]
                if self.isDial:
                    curextra = np.asarray([curextra],dtype = np.float32)

            if fin == len(self.x):
                self.curbatch = 0
                self.curdial+=1
                self.curdial %= self.nSamp
                self.x = np.load(self.embPath+'/X_'+str(self.dialIds[self.curdial]) +'.npy')
            else:
                self.curbatch+=1


        else:
            self.curbatch %= self.numBatches
            curx = self.x[self.curbatch * self.batchSize :(self.curbatch+1) * self.batchSize]
            cury = self.y[self.curbatch * self.batchSize :(self.curbatch+1) * self.batchSize]
            cursent = self.sentlen[self.curbatch * self.batchSize :(self.curbatch+1) * self.batchSize]
            curdial = self.dialLen[self.curbatch * self.batchSize :(self.curbatch+1) * self.batchSize]
            if self.extrafeats is not None:
                curextra = self.extrafeats[self.curbatch * self.batchSize :(self.curbatch+1) * self.batchSize]
            self.curbatch +=1
            taskdis = self.taskDiscTar


        return curx,cury,cursent,curdial,taskdis,curextra








