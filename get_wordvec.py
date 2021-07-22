import numpy as np
import multiprocessing
import jieba
import jieba.analyse

vac_dic = {}
with open("wvac.txt", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        l = line.split()
        vac_dic[l[0]] = np.array(l[1:]).astype(float)

uid, vid = {}, {}
j, k = 0, 0
with open("metadata", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        l = line.strip().replace("[","").replace("]","").split("\t")
        if l[1] not in uid:
            uid[l[1]] = j
            j += 1
        if l[2] not in vid:
            vid[l[2]] = k
            k += 1
print('uid & vid get')

class Reader(multiprocessing.Process):
    def __init__(self, file_name, emb_dict, prefix, start_pos, end_pos):
        multiprocessing.Process.__init__(self)
        self.file_name = file_name
        self.start_pos = start_pos
        self.end_pos = end_pos
        if prefix == "user":
            self.iddic = uid
        else:
            self.iddic = vid

    def run(self):
        fd = open(self.file_name, encoding="utf-8").readlines()
        lines = fd[self.start_pos:self.end_pos]
        pre_len = 10
        for ind, line in enumerate(lines):
            if ind% 10000 == 0:
                print(ind)
            l = line.strip().replace("WrappedArray","").replace("[","").replace("]","").split("\t")
            seg = "".join(l[1:]).replace("null","").replace("-100","").replace(" ","")
            wordwt = jieba.analyse.extract_tags(seg, topK=pre_len, withWeight=True, allowPOS=())
            tmp = np.zeros(64)
            for i in range(pre_len):
                if i < len(wordwt):
                    tmp += vac_dic.get(wordwt[i][0], vac_dic["</s>"])*wordwt[i][1]
            emb_dict[self.iddic[l[0]]] = tmp

class Partition(object):
    def __init__(self, file_name, thread_num):
        self.file_name = file_name
        self.block_num = thread_num
    def part(self):
        fd = open(self.file_name, encoding="utf-8").readlines()
        file_size = len(fd)
        pos_list = []
        block_size = file_size//self.block_num
        start_pos = 0
        for i in range(self.block_num):
            if i == self.block_num - 1:
                pos_list.append((start_pos, file_size))
                break
            end_pos = (i + 1) * block_size
            pos_list.append((start_pos, end_pos))
            start_pos = end_pos
        return pos_list

if __name__ == '__main__':
    emb_dict = multiprocessing.Manager().dict()
    #v_emb = multiprocessing.Manager().dict()
    thread_num = 16
    p = Partition("uemb", thread_num)
    t = []
    pos = p.part()
    for i in range(thread_num):
        t.append(Reader("uemb", emb_dict, "user", *pos[i]))
    for i in range(thread_num):
        t[i].start()
    for i in range(thread_num):
        t[i].join()

    print("get user emb file done")
    uemb_feat = []
    for i in range(len(uid)):
        uemb_feat.append(emb_dict[i])
    np.save("uemb-feats.npy", uemb_feat)
    print("uemb save done!")

    p = Partition("vemb", thread_num)
    t = []
    emb_dict = multiprocessing.Manager().dict()
    pos = p.part()
    for i in range(thread_num):
        t.append(Reader("vemb", emb_dict, "vendor", *pos[i]))
    for i in range(thread_num):
        t[i].start()
    for i in range(thread_num):
        t[i].join()

    vemb_feat = []
    for i in range(len(vid)):
        try:
            vemb_feat.append(emb_dict[i])
        except:
            vemb_feat.append(emb_dict[i-1])
    np.save("vemb-feats.npy", vemb_feat)
    print("vemb save done!")
