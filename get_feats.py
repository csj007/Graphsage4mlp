#!/usr/bin/env python3
## for python3 scripts
#coding:utf-8
import numpy as np
import tensorflow as tf
from sklearn import preprocessing as prepro

def get_norm_feats(input_npy):
    b = prepro.scale(input_npy)
    return b

def get_embedding(prefix, ckpt_path, X):
    graph = tf.Graph()
    with graph.as_default():
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        with sess.as_default():
            saver = tf.train.import_meta_graph(ckpt_path + ".meta")
            saver.restore(sess, ckpt_path)
            loss = graph.get_operation_by_name("dnn/hiddenlayer_2/batchnorm_2/batchnorm/add_1").outputs[0]
            x = graph.get_operation_by_name("dnn/input_from_feature_columns/input_layer/x_input/Reshape").outputs[0]
            re = sess.run(loss, {x:X})
            np.save(prefix + "_emb.npy",re)
        print(" Finished ")


def get_feat():
    oid, uid, vid = {}, {}, {}
    i, j, k = 0, 0, 0
    with open("metadata", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            l = line.strip().replace("[","").replace("]","").split("\t")
            if l[0] not in oid:
                oid[l[0]] = i
                i += 1
            if l[1] not in uid:
                uid[l[1]] = j
                j += 1
            if l[2] not in vid:
                vid[l[2]] = k
                k += 1

    print('uid & vid get')
    
    udic = {}
    with open("uf", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            l = line.strip().replace("[","").replace("]","").split("\t")
            udic[uid[l[0]]] = [float(x.replace('null', '0')) for x in l[-43:]]
    print("udic get")

    vdic = {}
    with open("vf", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            l = line.strip().replace("[","").replace("]","").split("\t")
            vdic[vid[l[0]]] = [float(x.replace('null', '0')) for x in l[-50:]]
    print("vdic get")
    
    odic = {}
    with open("of", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            l = line.strip().replace("[","").replace("]","").split("\t")
            odic[oid[l[0]]] = [float(x.replace('null', '0')) for x in l[-18:]]

    ufeats, vfeats, ofeats = [], [], []
    
    for i in range(len(uid)):
        try:
            ufeats.append(udic[i])
        except:
            ufeats.append(ufeats[i-1])
    for i in range(len(vid)):
        try:
            vfeats.append(vdic[i])
        except:
            vfeats.append(vfeats[i-1])
    
    for i in range(len(oid)):
        try:
            ofeats.append(odic[i])
        except:
            ofeats.append(odic[i-1])
    print("all feats get")

    ufeats=np.array(ufeats)
    vfeats=np.array(vfeats)
    ofeats=np.array(ofeats)
    ofeats[np.isnan(ofeats)]=0
    ufeats[np.isnan(ufeats)]=0
    vfeats[np.isnan(vfeats)]=0
    order_feats = get_norm_feats(ofeats)
    user_feats = get_norm_feats(ufeats)
    vendor_feats = get_norm_feats(vfeats)
    get_embedding("user", "runs/user_pre/model.ckpt-best", user_feats)
    get_embedding("vendor", "runs/vendor_pre/model.ckpt-best", vendor_feats)
    np.save("order-feats.npy", order_feats)
    print("all feats saved done!")


if __name__ == '__main__':
    get_feat()

