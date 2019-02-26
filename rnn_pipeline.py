import os, datetime, sys, json

import pandas as pd
import numpy as np
import tensorflow as tf

from constants import PROCESSED_PATH, RAW_PATH, DATA_PATH
from rnn_utils import *

learning_rate = 1e-4
max_train_steps = 10000
loss_thresh = 1e-4
display_step = 100
H, F, S = 24, 178, 4
batchSize = 75
seed = None
labelCol = 'STAGE'
labelStart = -2
idsPath = os.path.join(RAW_PATH, 'd_ids_split.pickle')
datapath = os.path.join(DATA_PATH, 'bypt_old')
savepath = os.path.join(DATA_PATH, 'model')

if len(sys.argv) > 1:
    with open(sys.argv[1], 'r') as cfgf:
        cfg = json.loads(cfgf.read())
        learning_rate = cfg.get('learning_rate', learning_rate)
        max_train_steps = cfg.get('max_train_steps', max_train_steps)
        loss_thresh = cfg.get('loss_thresh', loss_thresh)
        display_step = cfg.get('display_step', display_step)
        H, F, S = cfg.get('H_F_S', [H, F, S])
        batchSize = cfg.get('batchSize', batchSize)
        seed = cfg.get('seed', seed)
        labelCol = cfg.get('labelCol', labelCol)
        labelStart = cfg.get('labelStart', labelStart)
        idsPath = cfg.get('idsPath', idsPath)
        datapath = cfg.get('datapath', datapath)
        savepath = cfg.get('savepath', savepath)
        print('loaded config %s' % sys.argv[1])
        # print(cfg)


split_ids = getSplitIds(idsPath)
db = DataBatch(datapath, split_ids, batchSize=batchSize)
print([(k, len(v)) for k, v in db.files.items()])

trainBatches = db.getBatchIterator('devel')
testBatches = db.getBatchIterator('test')

def RNN(xs, batch_size):
    with tf.variable_scope("MyRNN"):
        LSTMcells = [tf.contrib.rnn.LSTMCell(s) for s in [F, S]]
        cell = tf.contrib.rnn.MultiRNNCell(LSTMcells)
        
#         LSTMcell = tf.contrib.rnn.LSTMCell(F)
#         MRcell = tf.contrib.rnn.MultiRNNCell([LSTMcell])
#         cell=tf.contrib.rnn.OutputProjectionWrapper(MRcell, output_size=S)
   
        initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        output, state = tf.nn.dynamic_rnn(cell, xs, initial_state=initial_state)
        return output

tf.reset_default_graph()
xs = tf.placeholder(shape=[None, H, F], dtype=tf.float32)
yt = tf.placeholder(shape=[None, H, S], dtype=tf.float32)
batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')
output = RNN(xs, batch_size)

loss = tf.reduce_mean(tf.nn.l2_loss(yt-output))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
train=optimizer.minimize(loss)

prediction = tf.nn.softmax(output)
precat = tf.argmax(prediction, 2)
labels = tf.argmax(yt, 2)
correct_pred = tf.equal(precat, labels)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

confmat = tf.confusion_matrix(
    labels=tf.reshape(labels, [-1]),
    predictions=tf.reshape(tf.argmax(prediction, 2), [-1])
)

saver = tf.train.Saver()

pro, pre, tru = None, None, None
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("output", sess.graph)
#     print(sess.run())
    row_nan_bool = None
    headers = None
    
    prev_lss = 1
    lss_change = None
    step = 1
    
    while (lss_change is None) or (lss_change > loss_thresh and step < max_train_steps):
        nb = 1

        # training
        for batch in trainBatches.setSeed(seed):
            # print('bnumnan', np.sum(np.isnan(batch)))
            trn_ids, trn_X, trn_Y = None, None, None
            if headers is None:
                headers = db.getHeaders()
                # batch, headers, row_nan_bool = dropNanCols(batch, headers, row_nan_bool)
                # print(headers)
                trn_ids, trn_X, trn_Y, hdrscut = prepareData(batch, headers, nclasses=4, labelCol=labelCol, labelStart=labelStart, debug=True) # 
                print(hdrscut)
            else:
                # batch, _, row_nan_bool = dropNanCols(batch, None, row_nan_bool)
                trn_ids, trn_X, trn_Y, hdrscut = prepareData(batch, headers, nclasses=4, labelCol=labelCol, labelStart=labelStart)

            # print('Batch', nb)
            nb += 1
            # Run optimization op (backprop)
            lss, _ = sess.run([loss, train], feed_dict={xs:trn_X,yt:trn_Y, batch_size:trn_X.shape[0]})
            # lss_change = abs(prev_lss-lss)/prev_lss
            lss_change = prev_lss-lss

            prev_lss = lss
            
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            (otp, acc,) = sess.run([output, accuracy],
                                   feed_dict={xs:trn_X, yt:trn_Y, batch_size:trn_X.shape[0]})

            # print('numnan', np.sum(np.isnan(otp)))
            print("Step %5d | L2 Loss = %.4f, Train Accuracy = %.3f" % (step, lss, acc))
        
        step += 1
        
    print("Optimization Finished!")
    trn_lss, trn_acc = sess.run([loss, accuracy], feed_dict={xs:trn_X, yt:trn_Y, batch_size:trn_X.shape[0]})
    print("Step %5d | L2 Loss = %.4f, Train Accuracy = %.3f" % (step, lss, acc))
    
    # testing
    for test in testBatches.setSeed(seed):  
        # test, _, _ = dropNanCols(batch, None, row_nan_bool)
        tst_ids, tst_X, tst_Y, _ = prepareData(test, headers, nclasses=4, labelCol=labelCol, labelStart=labelStart)
        prob, preb, trub = sess.run([output, precat, labels],
                                     feed_dict={xs:tst_X, yt:tst_Y, batch_size:tst_X.shape[0]})
        # print('numnan', np.sum(np.isnan(prob)))
        if pre is None:
            pro, pre, tru = prob, preb, trub
        else:
            pro = np.concatenate([pro, prob], axis=0) 
            pre = np.concatenate([pre, preb], axis=0)
            tru = np.concatenate([tru, trub], axis=0)
        
        cor = (pre == tru).flatten()
        print('Test accuracy:', np.sum(cor) / len(cor))

    mdir = datetime.datetime.now().strftime('%m%d%y%H%M%S')
    saver.save(sess, os.path.join(savepath, mdir, 'model_%s.ckpt' % mdir))
    writer.close()

metricsfn = os.path.join(PROCESSED_PATH, 'rnn_metrics_%s.csv' % datetime.datetime.now().strftime('%m%d%y%H%M%S'))
aucsfn = os.path.join(PROCESSED_PATH, 'rnn_aucs_%s.csv' % datetime.datetime.now().strftime('%m%d%y%H%M%S'))

saveMetrics(tru, pre, metricsfn)
saveAUCs(tru, pro, aucsfn)


