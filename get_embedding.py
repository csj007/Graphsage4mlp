#!/usr/bin/env python3
## for python3 scripts
from __future__ import division
from __future__ import print_function

import os
import gc
import sys
import numpy as np
import networkx as nx
import tensorflow as tf
from gsminibatch import NodeMinibatchIterator

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
flags.DEFINE_string('train_prefix', 'user', 'prefix identifying training data. must be specified.')
flags.DEFINE_string('ckpt_path', 'model/user/checkpoints/model-780', 'model path and name')
flags.DEFINE_integer('max_degree', 80, 'maximum node degree.')
flags.DEFINE_integer('batch_size', 1024, 'minibatch size.')
flags.DEFINE_integer('feed_batch_size', 1024, 'minibatch size.')
flags.DEFINE_integer('gpu', 0, "which gpu to use.")

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)


def prepare():
    with open("metadata", encoding="utf-8") as f:
        user2vendor = {}
        vendor2user = {}
        uid = {}
        vid = {}
        j, k = 0, 0
        lines = f.readlines()
        for line in lines:
            l = line.replace("[","").replace("]","").split("\t")
            if l[1] not in user2vendor:
                user2vendor[l[1]] = {}
            if l[2] not in vendor2user:
                vendor2user[l[2]] = {}
            if l[1] not in uid:
                uid[l[1]] = j
                j += 1
            if l[2] not in vid:
                vid[l[2]] = k
                k +=1
            if l[2] not in user2vendor[l[1]]:
                user2vendor[l[1]][l[2]] = 1
            if l[1] not in vendor2user[l[2]]:
                vendor2user[l[2]][l[1]] = 1
        u2v=user2vendor
        v2u=vendor2user
        nodedict = {}
        if FLAGS.train_prefix =='user':
            uid_new = dict(zip(uid.values(), uid.keys()))
            for i in range(len(uid)):
                if i%100000 == 0:
                    print(i)
                nodedict[i] = {}
                nodedict[i][uid_new[i]] = 0
                for v in u2v[uid_new[i]]:
                    if len(v2u[v])>100:
                        nodedict[i][uid_new[i]] += 100
                        continue
                    for u in v2u[v]:
                        if u not in nodedict[i]:
                            nodedict[i][u]=0
                        nodedict[i][u] += len(v2u[v])
        else:
            vid_new = dict(zip(vid.values(), vid.keys()))
            for i in range(len(vid)):
                if i%10000 == 0:
                    print(i)
                nodedict[i] = {}
                for u in v2u[vid_new[i]]:
                    for v in u2v[u]:
                        if v not in nodedict[i]:
                            nodedict[i][v]=0
                        nodedict[i][v] += len(u2v[u])
        return user2vendor, vendor2user, uid, vid, nodedict


def getidmap(id):
    idmap = {}
    for i in range(len(id)):
        idmap[i] = i
    return idmap

def getG_u(uid, u2v, userdict):
    uid_new = dict(zip(uid.values(), uid.keys()))
    G = nx.Graph()
    pt,nt, valn = 0, 0, 0
    for i in range(len(uid)):
        if i%100000 == 0:
            print(i)
            sys.stdout.flush()
        G.add_node(i)
        G.nodes[i]['val'] = True
        G.nodes[i]['test'] = True
        a = len(u2v[uid_new[i]])
        for u in userdict[i].keys():
            if uid[u]>i:
                b = len(u2v[u])
                Inter = userdict[i][u]
                Union_a = userdict[i][uid_new[i]]
                Union_b = userdict[uid[u]][u]
                U = 1/np.log(Union_a + a + 1) + 1/np.log(Union_b + b + 1)
                I = 1/np.log(Inter + a + 1) + 1/np.log(Inter + b + 1)
                if I/U<=1.1:
                    G.add_edge(i, int(uid[u]))
    return G

def getG_v(vid, vendordict, v2u):
    vid_new = dict(zip(vid.values(), vid.keys()))
    G = nx.Graph()
    for i in range(len(vid)):
        if i%10000 == 0:
            print(i)
            sys.stdout.flush()
        G.add_node(i)
        G.nodes[i]['val'] = True
        G.nodes[i]['test'] = True
        a = len(v2u[vid_new[i]])
        for v in vendordict[i].keys():
            if vid[v]>i:
                b = len(v2u[v])
                Union_a = vendordict[i][vid_new[i]]
                Union_b = vendordict[vid[v]][v]
                U = 1/np.log(Union_a + a + 1) + 1/np.log(Union_b + b + 1)
                Inter = vendordict[i][v]
                I = 1/np.log(Inter + a + 1) + 1/np.log(Inter + b + 1)
                if I/U<=1.3:
                    G.add_edge(i, int(vid[v]))
    return G

def load_data(G, id_map, normalize=True):
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n
    print("Loaded feats... now convert to numpy")
    feats = np.load(FLAGS.train_prefix + "-feats.npy")
    print("Done feats... now handle id_map")
    sys.stdout.flush()
    id_map = {conversion(k):int(v) for k,v in id_map.items()}

    print("Normalize feats...")
    sys.stdout.flush()
    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() ])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    return G, feats, id_map

def construct_placeholders():
    # Define placeholders
    placeholders = {
        'batch' : tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
        'adj': tf.placeholder(tf.int32, shape=(None, FLAGS.max_degree),name='adj'),
    }
    return placeholders

def train(train_data, test_data=None):
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]

    print("Loaded all data... now ready to test...")
    sys.stdout.flush()
    placeholders = construct_placeholders()
    minibatch = NodeMinibatchIterator(G,
            id_map,
            placeholders,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree)

    graph = tf.Graph()
    with graph.as_default():
        config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        config.gpu_options.allow_growth = True
        
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        with sess.as_default():
            finished = False
            iter_num = 0
            saver = tf.train.import_meta_graph("{}.meta".format(FLAGS.ckpt_path))
            saver.restore(sess, FLAGS.ckpt_path)
            outs = graph.get_operation_by_name("l2_normalize").outputs[0]
            bs = graph.get_operation_by_name("batch_size").outputs[0]
            bt = graph.get_operation_by_name("batch1").outputs[0]
            adj = graph.get_operation_by_name("adj").outputs[0]
            feats = graph.get_operation_by_name("Variable").outputs[0]
            try:
                feats_op = tf.assign(feats, features, validate_shape=False)
                res = {}
                sess.run(feats_op)
                del features
                gc.collect()
                while not finished:
                    batch_size, batch1, finished, nodes = minibatch.incremental_embed_feed_dict(FLAGS.feed_batch_size, iter_num)
                    print("iter_num is " + str(iter_num))
                    sys.stdout.flush()
                    iter_num += 1
                    out = sess.run(outs, {bs:batch_size, bt:batch1, adj:minibatch.adj})
                    for i, node in enumerate(nodes):
                        if node not in res:
                            res[node]=out[i,:].tolist()
                feat_data = []
                for i in range(len(res)):
                    feat_data.append(res[i])
            except:
                feats = np.load(FLAGS.train_prefix+"_emb.npy")
                out = np.concatenate((feats.T, feats.T))
                feat_data = out.T
            np.save(FLAGS.train_prefix + "_graphemb.npy", feat_data)
            print("Get " + FLAGS.train_prefix + " Embedding Done!")
            sys.stdout.flush()

def main(argv=None):
    print('prepare necessary file')
    sys.stdout.flush()
    u2v, v2u, uid, vid, nodedict = prepare()
    if FLAGS.train_prefix == 'user':
        print('make id file')
        sys.stdout.flush()
        idmap = getidmap(uid)
        print('make user G file')
        sys.stdout.flush()
        G = getG_u(uid, u2v, nodedict)
    elif FLAGS.train_prefix == 'vendor':
        print('make id file')
        sys.stdout.flush()
        idmap = getidmap(vid)
        print('make vendor G file')
        sys.stdout.flush()
        G = getG_v(vid, nodedict, v2u)
    else:
        "Wrong train prefix"
    print('Build Graph Data Done!')
    sys.stdout.flush()
    print("Loading training data..")
    sys.stdout.flush()
    train_data = load_data(G, idmap)
    del G, idmap, u2v, v2u, uid, vid, nodedict
    gc.collect()
    print("Done loading training data..")
    sys.stdout.flush()
    train(train_data)

if __name__ == '__main__':
    tf.app.run()

