import os
os.system('pip install jieba==0.39')
import gc
import time
import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('model_loc', '', 'Local graph model.')
tf.flags.DEFINE_string('output_loc', '', 'output dir.')
tf.flags.DEFINE_string('out_name', '', 'output file name.')
tf.flags.DEFINE_string('ckpt_path', '', 'output file name.')
tf.flags.DEFINE_string('dt', '', 'now date')
tf.flags.DEFINE_string('data_dt', '', 'calculate date')
tf.flags.DEFINE_string('clear_dt', '', 'clear date')
tf.flags.DEFINE_string('data_path', '', 'data path')
tf.flags.DEFINE_string('db', '', 'database')


def concat():
    a = np.load("user_emb.npy")
    b = np.load("uemb-feats.npy")
    c = np.load("vendor_emb.npy")
    d = np.load("vemb-feats.npy")
    udata = np.concatenate((a.T, b.T))
    vdata = np.concatenate((c.T, d.T))
    np.save("user-feats.npy", udata.T)
    np.save("vendor-feats.npy", vdata.T)


def evaluate_order():
    print("Concatenate Order Features...")
    feats = []
    ofeats = np.load("order-feats.npy")
    ufeats = np.load("user_graphemb.npy")
    vfeats = np.load("vendor_graphemb.npy")
    oid = {}
    uid = {}
    vid = {}
    i, j, k = 0, 0, 0
    print("All file loaded...now preprocessing...")
    with open("metadata", encoding="utf-8") as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if index%100000 == 0:
                print(index)
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
            tmp = np.concatenate((ufeats[uid[l[1]]], vfeats[vid[l[2]]], ofeats[oid[l[0]]]))
            feats.append(tmp)
    X = np.array(feats)
    print("Order feats concated... now testing...")
    del feats, ofeats, ufeats, vfeats
    gc.collect()

    graph = tf.Graph()
    with graph.as_default():
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        with sess.as_default():
            saver = tf.train.import_meta_graph(FLAGS.ckpt_path + ".meta")
            saver.restore(sess, FLAGS.ckpt_path)
            outs = graph.get_operation_by_name("Sigmoid_1").outputs[0]
            x = graph.get_operation_by_name("x-input").outputs[0]
            re = sess.run(outs, {x:X})
            re = np.array(re)
        print("Testing Finished!\n")
    print("Start to store result!\n")
    lines = open("metadata", encoding="utf-8").readlines()
    with open(FLAGS.out_name + ".txt", "w") as f:
        for index, line in enumerate(lines):
            if index%100000 == 0:
                print(index)
            l = line.strip().replace("[","").replace("]","").split("\t")
            f.write(l[0] + "\t" + str(float(re[oid[l[0]]])) + "\n")

def save_uv_emb():
    user_graphemb = np.load("user_graphemb.npy")
    vendor_graphemb = np.load("vendor_graphemb.npy")
    kepp_precision = 4
    uid, vid = {}, {}
    j, k = 0, 0
    user_file = open("user_out_emb.txt", "w", encoding="utf-8")
    vendor_file = open("vendor_out_emb.txt", "w", encoding="utf-8")
    with open("metadata", encoding="utf-8") as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            l = line.strip().replace("[","").replace("]","").split("\t")
            if l[1] not in uid:
                uid[l[1]] = j
                useremb_list=np.around(user_graphemb[j], decimals=kepp_precision, out=None).tolist()
                useremb_str = ",".join(str(x) for x in useremb_list)
                user_file.write(l[1]+"\t"+useremb_str+'\n')
                j += 1
            if l[2] not in vid:
                vid[l[2]] = k
                vendoremb_list=np.around(vendor_graphemb[k], decimals=kepp_precision, out=None).tolist()
                vendoremb_str = ",".join(str(x) for x in vendoremb_list)
                vendor_file.write(l[2]+"\t"+vendoremb_str+'\n')
                k += 1
    user_file.close()
    vendor_file.close()

if __name__ == '__main__':
    try:
        os.system('hadoop fs -get ' + FLAGS.data_path + 'graph_data/' + FLAGS.data_dt + '/metadata/*.txt metadata')
        os.system('hadoop fs -get ' + FLAGS.data_path + 'graph_data/' + FLAGS.data_dt + '/userfeats/*.txt uf')
        os.system('hadoop fs -get ' + FLAGS.data_path + 'graph_data/' + FLAGS.data_dt + '/orderfeats/*.txt of')
        os.system('hadoop fs -get ' + FLAGS.data_path + 'graph_data/' + FLAGS.data_dt + '/vendorfeats/*.txt vf')
        os.system('hadoop fs -get ' + FLAGS.data_path + 'graph_data/' + FLAGS.data_dt + '/userword/*.txt uemb')
        os.system('hadoop fs -get ' + FLAGS.data_path + 'graph_data/' + FLAGS.data_dt + '/vendorword/*.txt vemb')
        os.system('hadoop fs -get ' + FLAGS.data_path + 'wvac.txt ./')
        os.system('hadoop fs -rm ' + FLAGS.output_loc + '*')
        os.system('hadoop fs -rm ' + FLAGS.data_path + 'user_out_emb.txt')
        os.system('hadoop fs -rm ' + FLAGS.data_path + 'vendor_out_emb.txt')
        os.system('hadoop fs -get ' + FLAGS.model_loc + '*')
        os.system('python get_feats.py')
        os.system('python get_wordvec.py')
        concat()
        os.system('rm -f uf vf of uemb vemb wvac.txt')
        os.system('sh run_get_embedding.sh')
        time.sleep(300)
        while True:
            if os.path.exists('user_graphemb.npy'): #and os.path.exists('vendor_graphemb.npy'):
                time.sleep(120)
                break
            else:
                print('waite for user_graphemb.npy...')
                time.sleep(120)
        evaluate_order()
        os.system('hadoop fs -put ' + FLAGS.out_name + '.txt ' + FLAGS.output_loc)
    except:
        print("------------------run code Error!-----------------")
    finally:
        print('ALL DONE!')
