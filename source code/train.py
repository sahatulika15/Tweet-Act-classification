import tensorflow as tf
import numpy as np
from attention import attention
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import LSTMCell 
from tensorflow.python.framework import ops
import pickle as pkl
import batGenBert 
from batGenBert import BatchGenerator
from sklearn.metrics import accuracy_score
import sys
import random

class FlipGradientBuilder(object):
    '''Gradient Reversal Layer from https://github.com/pumpikano/tf-dann'''

    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y

flip_gradient = FlipGradientBuilder()


def diff_loss(shared_feat, task_feat):
    '''Orthogonality Constraints from https://github.com/tensorflow/models,
    in directory research/domain_adaptation
    '''
    task_feat -= tf.reduce_mean(task_feat, 0)
    shared_feat -= tf.reduce_mean(shared_feat, 0)

    task_feat = tf.nn.l2_normalize(task_feat, 1)
    shared_feat = tf.nn.l2_normalize(shared_feat, 1)

    correlation_matrix = tf.matmul(
        task_feat, shared_feat, transpose_a=True)

    cost = tf.reduce_mean(tf.square(correlation_matrix)) * 0.01
    cost = tf.where(cost > 0, cost, 0, name='value')

    assert_op = tf.Assert(tf.is_finite(cost), [cost])
    with tf.control_dependencies([assert_op]):
        loss_diff = tf.identity(cost)

    return loss_diff

def shareLay(x,seqlen):
    shareConvOut = tf.layers.conv1d(inputs = x, filters = 64, kernel_size = [5],strides=1,padding='same',activation = tf.nn.relu, name='SharedConv')
    rnn_fw = LSTMCell(90)
    rnn_bw = LSTMCell(90)

    outSha, (stSha,_) = bi_rnn(rnn_fw, rnn_bw, inputs=shareConvOut, dtype=tf.float32,sequence_length= seqlen) ####################################################### # sequence_length=seq_len_ph,
    shareRnn_out = tf.concat(stSha,-1)
    return shareRnn_out


def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)


def adv_loss(attnout, label, keep_prob=0.5):

    attnout = flip_gradient(attnout)
    # num_task = self.maxTask
    drop = tf.nn.dropout(attnout, keep_prob)
    dense1 = tf.layers.dense(inputs=drop, units=30, activation=tf.nn.relu)
    share_y_hat = tf.layers.dense(inputs=dense1, units=totTaskNum)
    share_y_arg = tf.argmax(share_y_hat, axis=1)
    # task_label = tf.one_hot(label, self.maxTask, axis=-1)
    #label2 = tf.reshape(label,shape = [tf.shape()])
    loss_adv = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=share_y_hat))
    return loss_adv

def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    epsilon = 1e-9
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = tf.multiply(scalar_factor , vector,name='sqRes')  # element-wise
    return(vec_squashed)

def crfLoss(capVecLen, inpt, target, dial_lengths):
    crfIn = tf.reshape(capVecLen, shape=[tf.shape(inpt)[0], tf.shape(inpt)[1], capVecLen.get_shape()[-1]])
    targetInds = tf.argmax(target, axis=2, name='targetLabels')
    logLike, trParams = tf.contrib.crf.crf_log_likelihood(
        crfIn, targetInds, dial_lengths)
    cost = 100*tf.reduce_mean(-logLike)
    acc = tf.zeros([1],dtype=tf.float32)
    return cost,acc,crfIn,trParams

def capLoss(capVecLen, target, sentLen):
    max_l = tf.square(tf.maximum(0., 0.9 - capVecLen))
    max_r = tf.square(tf.maximum(0., capVecLen - 0.1))
    capstar = tf.reshape(target, [tf.shape(target)[0] * tf.shape(target)[1], target.get_shape()[2]])
    L_c = capstar * max_l + 0.5 * (1 - capstar) * max_r
    Lc = tf.reduce_sum(L_c, axis=1)
    mask = tf.cast(tf.greater(sentLen, 0), tf.float32)
    num_nz = tf.reduce_sum(mask)
    loss = tf.reduce_sum(Lc * mask) / num_nz
    pred = tf.argmax(capVecLen, axis=1)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, tf.argmax(capstar,1)), tf.float32))
    trparams = tf.zeros([1], dtype=tf.float32)
    return loss,acc,pred,trparams,tf.argmax(capstar,1)

def svmLoss(W,capVecLen,target,penalty):
    regLoss = tf.reduce_mean(tf.square(W))
    hinge_loss = tf.reduce_mean(tf.squa)

def crEntLoss(capVecLen,target):
    print('USING CENT LOSS')
    capstar = tf.reshape(target, [tf.shape(target)[0] * tf.shape(target)[1], target.get_shape()[2]])
    pred = tf.argmax(tf.nn.softmax(capVecLen),axis = 1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = capstar, logits=capVecLen))
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, tf.argmax(capstar,1) ), tf.float32))
    trparams = tf.zeros([1], dtype=tf.float32)
    return loss,acc,pred,trparams,tf.argmax(capstar,1)

def addExtra(extrafeats,toconcat):
    exshape = tf.shape(extrafeats)
    extradesnsein = tf.reshape(extrafeats, [exshape[0] * exshape[1], 4110])
    featdense = tf.layers.dense(inputs=extradesnsein, units=100, activation=tf.nn.relu, name='densefeats')
    featdenseresh = tf.reshape(featdense, [tf.shape(extradesnsein)[0], 100,1])
    toret = tf.concat((toconcat,featdenseresh),axis = 1)
    return toret


inp = []
sentLen0 = []
sentLen = []
target = []
extrafeats_ph = []
extrafeats = {}
tasklabel = []
dial_lengths = []
# inpShape = []
# cnnInpShape = []
convin = []
convOut = []
privRnnOut = []
shRnnOut = []
rnn_out = []
convin2 = []
capsin = []
capsules = []
squashin = []
sqOut = []
w_sh = []
b_IJ = []
W = []
biases = []
routeIn = []
u_hat = []
u_hat_reshape = []
u_hat_stopped = []
c_IJ = []
s_J = []
capOut = []
capOutVecNorm  = []
ltype = ['caps','caps']
lossAdv = []
diffLoss = []
ceLoss = []
acc = []
trParaMAt = []
pred = []
loss = []
g_step = []
opt = []
train_acc_ph = []
train_loss_ph = []
train_grad_ph = []
summary_acc = []
summary_loss = []
summary_grad = []
summary_op = []
gradobs = []
summary_grad = []


mxDial = 5
mxWords = 128
nEmb = 768
totTaskNum = 1
privAttnSize = 50
nClasses = [7,5]
nIter = 4
epsilon = 1e-9
capVecLen = 32
numCap = 20
outVecLen = 16
vocab_size = 24290
decaystep = [45000,45000]
cuslr = [1.5e-5,1e-5]
twChoose = 0.5
hasextra = [True,False]

print('CONFIG:  nIter:' ,nIter,"| capVecLen:",capVecLen,'| numCap:',numCap ,'| OutVecLen: ',outVecLen)
print('IN {} Diff loss by 10 and adv by 0.1, initla {} and decay {} and the prob of choosing tweeet is 1.0 \n LSTM:90 '.format(sys.argv[0], cuslr,decaystep))
print('Remark: bert caps lstm cnn with capsLoss ')
# with open('Data/emb_mat_sw_tw.pkl','rb') as fd: 
#     emb_mat = pkl.load(fd,encoding='bytes') 

with tf.variable_scope('SharedEnc', reuse=tf.AUTO_REUSE) as sc:
    pass
with tf.variable_scope('SharedADV', reuse=tf.AUTO_REUSE) as scADV:
    pass

# with tf.variable_scope('lookup', reuse=tf.AUTO_REUSE) as embsc:
#     inp_ph = tf.placeholder(tf.int32, shape = (None,None,mxWords), name='inp_ph')
#     lut_ph = tf.placeholder(tf.float32,shape = [vocab_size,nEmb],name = 'lut_ph')
#     lut_ph2 = tf.Variable(lut_ph)
#     embinp = tf.nn.embedding_lookup(lut_ph2,inp_ph,name = 'emplu')

tasknums = [i for i in range(totTaskNum)]


for i in tasknums:
    with tf.variable_scope('TaskNo' + str(i)) as curSc:
        inp.append(tf.placeholder(tf.float32, shape = [None,None,mxWords,nEmb], name='inp'+str(i)))
        inpShape = tf.shape(inp[i])
        sentLen0.append(tf.placeholder(tf.int32, shape=[None,None], name='numWrdsUtterance' + str(i)))
        sentLen.append(tf.reshape(sentLen0[i],shape = [inpShape[0] * inpShape[1]]))
        target.append(tf.placeholder(shape=[None,None,nClasses[i]], dtype=tf.float32, name='TargetTask'))
        extrafeats_ph.append(tf.placeholder(shape=[None, None, 4110], dtype=tf.float32, name='extraFeats_ph'))
        tasklabel.append(tf.placeholder(shape=[None,totTaskNum], dtype=tf.float32, name='TaskIDS'))
        dial_lengths.append(tf.placeholder(tf.int32, shape=[None],name='num_dial'))

        
        cnnInpShape = [inpShape[0] *inpShape[1] , inp[i].get_shape()[2], inp[i].get_shape()[3]]
        convin.append(tf.reshape(inp[i],cnnInpShape,name='conv1in'))
        
        convOut.append(tf.layers.conv1d(inputs=convin[i], filters=64, kernel_size=[5], strides=1, padding='same',
                                   activation=tf.nn.relu, name='conv1'))
        out, (st,_) = bi_rnn(LSTMCell(num_units=90), LSTMCell(num_units=90), inputs=convOut[i], dtype=tf.float32,
                         sequence_length=sentLen[i])
        privRnnOut.append(tf.concat(st, -1, name='PrivRnn_out'))
        #print(st)
        with tf.variable_scope(sc) as scpe:
            shRnnOut.append(shareLay(convin[i],sentLen[i]))
        #print(shRnnOut[i])
        convin_dum = tf.stack([privRnnOut[i], shRnnOut[i]], axis=-1,name='conv2In')
        print(convin_dum)
        convin2 = tf.expand_dims(privRnnOut[i],axis=-1) #tf.stack([privRnnOut[i], shRnnOut[i]], axis=-1,name='conv2In')
        print(convin2)
        #capsin.append(tf.expand_dims(convin2, axis=2, name='capsIn'))

        if hasextra[i]:
            convin2 = addExtra(extrafeats_ph[i], convin2)
            # with tf.variable_scope('extraFeatures'):

        capsin.append(convin2)
        #print(capsin[i])
        capsules.append(tf.layers.conv1d(inputs=capsin[i], filters=capVecLen * numCap, kernel_size=[9], strides=3,
                                    padding='valid', activation=tf.nn.relu))
        squashin = tf.reshape(capsules[i], [tf.shape(capsules[i])[0], numCap * capsules[i].get_shape()[1], capVecLen, 1],
                              name='squashIn')
        sqOut.append(squash(squashin))
        wsh = [1, sqOut[i].get_shape()[1].value, nClasses[i] * outVecLen, sqOut[i].get_shape()[-2].value,
               sqOut[i].get_shape()[-1].value]
        b_IJ = tf.zeros([tf.shape(sqOut[i])[0], sqOut[i].shape[1].value, nClasses[i], 1, 1], dtype=tf.float32, name='b_ij')
        W.append(tf.Variable(tf.random_normal(shape=wsh, stddev=0.1, dtype=tf.float32), dtype=tf.float32,name='W_inCaps'))

        biases = tf.Variable(tf.zeros((1, 1, nClasses[i], outVecLen, 1), dtype=tf.float32), dtype=tf.float32,name='Bias_inCaps')
        routeIn = tf.tile(tf.expand_dims(sqOut[i], axis=2), [1, 1, outVecLen * nClasses[i], 1, 1])
        u_hat = tf.reduce_sum(W[i] * routeIn, axis=3, keep_dims=True)
        u_hat_reshape.append(tf.reshape(u_hat, shape=[-1, wsh[1], nClasses[i], outVecLen, 1], name='u_hat_AlgoIn'))
        u_hat_stopped.append(tf.stop_gradient(u_hat_reshape[i], name='uhat_stop_gradient'))
        for r_iter in range(nIter):
            with tf.variable_scope('iter_' + str(r_iter)):
                c_IJ = tf.nn.softmax(b_IJ, axis=2)

                if r_iter == nIter - 1:
                    # weighting u_hat with c_IJ, element-wise in the last two dims
                    # => [batch_size, 1152, 10, 16, 1]
                    s_J = tf.multiply(c_IJ, u_hat_reshape[i])
                    # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                    s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
                    v_J = squash(s_J)

                elif r_iter < nIter - 1:
                    s_J = tf.multiply(c_IJ, u_hat_stopped[i])
                    s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
                    v_J = squash(s_J)
                    v_J_tiled = tf.tile(v_J, [1, wsh[1], 1, 1, 1])
                    u_produce_v = tf.reduce_sum(u_hat_stopped[i] * v_J_tiled, axis=3, keepdims=True)
                    b_IJ += u_produce_v

        capOut.append(tf.squeeze(v_J, axis=[1, 4], name='CapOutVector'))
        capOutVecNorm.append(tf.sqrt(tf.reduce_sum(tf.square(capOut[i]),
                                              axis=2, keepdims=False) + epsilon))
        
        

        with tf.variable_scope(scADV) as vsc:
            lossAdv.append(adv_loss(shRnnOut[i], tasklabel[i]))

        diffLoss.append(diff_loss(shRnnOut[i],privRnnOut[i]))
        act_tar= tf.zeros([1], dtype=tf.float32)
        with tf.variable_scope('taskloss'):
            if ltype[i] == 'crf':
                loss_CE, accu,preds, trParams = crfLoss(capOutVecNorm[i],inp[i],target[i],dial_lengths[i])
            elif ltype[i] == 'ce':
                loss_CE, accu,preds, trParams,act_tar = crEntLoss(capOutVecNorm[i],target[i])
            else:
                loss_CE, accu, preds, trParams,act_tar = capLoss(capOutVecNorm[i], target[i], sentLen[i])
        ceLoss.append(loss_CE)
        acc.append(accu)
        pred.append(preds)
        trParaMAt.append(trParams)
        #loss.append(tf.add_n([ceLoss[i], 10*diffLoss[i], 0.1 * lossAdv[i]]))
        loss.append(ceLoss[i])

        g_step.append(tf.Variable(0, name="tr_global_step", trainable=False))
        learning_rate = tf.train.exponential_decay(cuslr[i], g_step[i],decaystep[i], 0.5, staircase=True)
        opt.append(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss[i],global_step=g_step[i]))
        gradobs.append(tf.reduce_mean(tf.gradients(loss[i],capsin[i])))

        train_acc_ph.append(tf.placeholder(tf.float32,name='ac_ph'))
        train_loss_ph.append(tf.placeholder(tf.float32,name='loss_ph'))
        train_grad_ph.append(tf.placeholder(tf.float32,name='grad_ph'))
        summary_acc.append(tf.summary.scalar('ACC_' + str(i), train_acc_ph[i]))
        summary_loss.append(tf.summary.scalar('LOSS_' + str(i), train_loss_ph[i]))
        summary_grad.append(tf.summary.scalar('grad_' + str(i), train_grad_ph[i]))
        summary_op.append(tf.summary.merge([summary_loss[i], summary_acc[i], summary_grad[i]]))
        

saver = tf.train.Saver(max_to_keep=100)
savedir = ''
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)

numEpoch = 100
maxbatch = 32 # maximum no of batches in all given dataset included
numbatch = []
batchSize = [32,32]

#paths = ['../Data/swbd_trainset_bert.pkl','../Data/twit_trainset_bert.pkl']
#embPaths = ['../../../Data_bert/swbd_tr', '../../../Data_bert/tw_tr']
#embValidPaths = ['../../../Data_bert/swbd_te', '../../../Data_bert/tw_tr']
#validpaths = ['../Data/swbd_validset_bert.pkl','../Data/twit_validset_bert.pkl']


paths = ['Data/swbd_trainset_bert.pkl','Data/twit_trainset_bert.pkl']
embPaths = ['../../Data_bert/swbd_tr', '../../Data_bert/tw_tr']
embValidPaths = ['../../Data_bert/swbd_te', '../../Data_bert/tw_tr']
validpaths = ['Data/swbd_validset_bert.pkl','Data/twit_validset_bert.pkl']

paths.reverse()
embPaths.reverse()# = ['../../Data_bert/swbd_tr', '../../Data_bert/tw_tr']
embValidPaths.reverse()#  = ['../../Data_bert/swbd_te', '../../Data_bert/tw_tr']
validpaths.reverse()# = ['Data/swbd_validset_bert.pkl','Data/twit_validset_bert.pkl']

bg = []
isdial = [True,True]
for i in range(totTaskNum):
    bgen = BatchGenerator(paths[i],embPaths[i],batchSize[i],i,totTaskNum,isdial[i],hasextra[i],False)
    bg.append(bgen)
numbatch = [bgn.numBatches for bgn in bg]

maxbatch = max([bgn.numBatches for bgn in bg])
# numbatch = [bgn.numBatches for bgn in bg]

print("number of Batches:",numbatch)
sys.stdout.flush()
modelver = 'bestModel'

tf.set_random_seed(786)
np.random.seed(786)
random.seed(786)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer()) # ,feed_dict={lut_ph:emb_mat}
    summary_writer = tf.summary.FileWriter('./tfl2/task1/baselines/' + '/train_' + modelver, sess.graph)
    summary_writer.add_graph(sess.graph)

    best_vacc = -1
    for epno in range(numEpoch):
        epLoss = {}
        epAcc = {}
        # for each epoch calc avg loss and accuracy
        for tasks in range(totTaskNum):
            epLoss[tasks] = 0.0
            epAcc[tasks] = 0.0

        for batches in range(maxbatch): ##############################3 50
            for task_id in range(totTaskNum):
                if task_id == 1 and batches != maxbatch-1:
                    if random.uniform(0,1) > twChoose:
                        continue        
                lvec_bat = 0
                x_batch,y_batch,sent_batch,dial_batch,taskdisc,exfeat = bg[task_id].nextBatch()
                #x_batch = sess.run(embinp,feed_dict={inp_ph:x_batch})
                if hasextra[task_id]:
                    l_bat, a_bat, pre_bat, st_bat, para_bat, grad_bat, _ = sess.run(
                        [loss[task_id], acc[task_id], pred[task_id], g_step[task_id], trParaMAt[task_id],
                         gradobs[task_id], opt[task_id]], feed_dict={inp[task_id]: x_batch,
                                                                     sentLen0[task_id]: sent_batch,
                                                                     target[task_id]: y_batch,
                                                                     tasklabel[task_id]: taskdisc,
                                                                     dial_lengths[task_id]: dial_batch,
                                                                     extrafeats_ph[task_id]:exfeat
                                                                     })
                else:
                    l_bat,a_bat,pre_bat,st_bat,para_bat,grad_bat,_ = sess.run([loss[task_id],acc[task_id],pred[task_id],g_step[task_id],trParaMAt[task_id],gradobs[task_id],opt[task_id] ], feed_dict={inp[task_id]: x_batch,
                                                                                                                                    sentLen0[task_id]: sent_batch,
                                                                                                                                    target[task_id] : y_batch ,
                                                                                                                                    tasklabel[task_id]: taskdisc,
                                                                                                                                    dial_lengths[task_id]: dial_batch
                                                                                                                                    })
                        
                #print(pre_bat[:10],tar_bat[:10])
                if ltype[task_id] == 'crf':
                    vit_pred = []
                    vit_true = []
                    for i in range(len(pre_bat)):
                        dum1,_ =  tf.contrib.crf.viterbi_decode(pre_bat[i][:dial_batch[i]],para_bat)
                        vit_pred += dum1[:dial_batch[i]]
                        vit_true += y_batch[i][:dial_batch[i]].tolist()

                    vit_pred = np.asarray(vit_pred,dtype= np.int32)
                    vit_true = np.argmax(np.asarray(vit_true,dtype= np.int32),axis=1)
                    a_bat = accuracy_score(vit_true,vit_pred)
                    #print(vit_pred[:10],vit_true[:10])
                summary = sess.run(summary_op[task_id],feed_dict={train_acc_ph[task_id]:a_bat,
                                                                        train_loss_ph[task_id]:l_bat,
                                                                        train_grad_ph[task_id]:grad_bat})
                
                summary_writer.add_summary(summary,global_step=st_bat)

        if epno % 1 == 0:
            #savepath = 'Models/' + modelver + '/' +'epoch' + str(epno + 1)
            #saver.save(sess,savepath)

            for task_id in range(totTaskNum):
                VAcc = 0

                vBgen = BatchGenerator(validpaths[task_id],embValidPaths[task_id],batchSize[task_id],task_id,totTaskNum,isdial[task_id],hasextra[task_id],False)
                for batches in range(vBgen.numBatches):
                    x_batch, y_batch, sent_batch, dial_batch, taskdisc,exfeat = vBgen.nextBatch()
                    #x_batch = sess.run(embinp,feed_dict={inp_ph:x_batch})
                    if hasextra[task_id]:
                        a_bat, pre_bat, para_bat = sess.run([acc[task_id], pred[task_id], trParaMAt[task_id]],feed_dict={inp[task_id]: x_batch,
                                                                                                                         target[task_id] : y_batch,
                                                                                                                         sentLen0[task_id]: sent_batch,
                                                                                                                         tasklabel[task_id]: taskdisc,
                                                                                                                         dial_lengths[task_id]: dial_batch,
                                                                                                                         extrafeats_ph[task_id]:exfeat
                                                                                                                         })
                    else:
                        a_bat, pre_bat, para_bat = sess.run([acc[task_id], pred[task_id], trParaMAt[task_id]],
                                                            feed_dict={inp[task_id]: x_batch,
                                                                       target[task_id]: y_batch,
                                                                       sentLen0[task_id]: sent_batch,
                                                                       tasklabel[task_id]: taskdisc,
                                                                       dial_lengths[task_id]: dial_batch
                                                                       })

                    if ltype[task_id] == 'crf':
                        vit_pred = []
                        vit_true = []
                        for i in range(len(pre_bat)):
                            dum1,_ = tf.contrib.crf.viterbi_decode(pre_bat[i][:dial_batch[i]], para_bat)
                            vit_pred += dum1
                            vit_true += y_batch[i][:dial_batch[i]].tolist()

                        vit_pred = np.asarray(vit_pred, dtype=np.int32)
                        vit_true = np.argmax(np.asarray(vit_true, dtype=np.int32),axis = 1)
                        a_bat = accuracy_score(vit_true, vit_pred)
                    VAcc+=a_bat
                VAcc/= vBgen.numBatches
                print('Epoch: {} | Task: {} | Accuracy {} \n'.format(epno,task_id,VAcc))
                if best_vacc < VAcc:
                    best_vacc = VAcc
                    best_epoch = epno
                    savepath = 'Models/task1/baselines/' + modelver + '/' +'best_epoch' + str(epno + 1)
                    saver.save(sess,savepath)
                sys.stdout.flush()
                del vBgen
        if epno % 2 == 3:
            savepath = 'Models/task1/baselines/' + modelver + '/' +'epoch' + str(epno + 1)
            saver.save(sess,savepath)
    print('best: {} : {}'.format(best_epoch,best_vacc))    
            



