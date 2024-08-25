import os
import json
import pickle
import numpy as np
import random
random.seed(666)


data_list=[
['WMT14','train.de','train.en','vocab.50K.de','vocab.50K.en'],
]


max_len=32


def check(word_list,vocab):
    unk_num=0
    for word in word_list:
        if word not in vocab:
            unk_num+=1
    if unk_num/len(word_list)>0.5:
        return 0
    else:
        return 1

for data_name,src_name,tgt_name,src_vocab,tgt_vocab in data_list:
    print(data_name)
    prefix=data_name+'/raw/'
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    fout1=open(prefix+'src_train.txt','w')
    fout2=open(prefix+'src_val.txt','w')
    fout_test_src=open(prefix+'src_test.txt','w')
    fout3=open(prefix+'tgt_train.txt','w')
    fout4=open(prefix+'tgt_val.txt','w')
    fout_test_tgt = open(prefix + 'tgt_test.txt', 'w')

    fin1=open(data_name+'/'+src_name,'r',encoding='ISO-8859-1')
    src_data=fin1.readlines()
    fin2=open(data_name+'/'+tgt_name,'r',encoding='ISO-8859-1')
    tgt_data=fin2.readlines()

    tot_data=list(zip(src_data,tgt_data))
    random.shuffle(tot_data)
    src_data=[x[0] for x in tot_data]
    tgt_data = [x[1] for x in tot_data]
    src_data,tgt_data=src_data[:1000000],tgt_data[:1000000]

    fin1=open(data_name+'/'+src_vocab,'r',encoding='ISO-8859-1')
    src_vocab_data=fin1.readlines()
    fin2=open(data_name+'/'+tgt_vocab,'r',encoding='ISO-8859-1')
    tgt_vocab_data=fin2.readlines()


    vocab1,vocab2={},{}
    ind=1
    for line in src_vocab_data:
        vocab1[line.rstrip('\r\n')]=ind
        ind+=1
    ind=1
    for line in tgt_vocab_data:
        vocab2[line.rstrip('\r\n')]=ind
        ind+=1
    with open(prefix+'src_vocab.pkl','wb') as f:
        pickle.dump(vocab1,f)
    with open(prefix+'tgt_vocab.pkl','wb') as f:
        pickle.dump(vocab2,f)
    train_num=int(len(src_data)*0.8)
    test_num=10000
    ind=0
    tot_data=list(zip(src_data,tgt_data))
    for i in range(len(tot_data)):
        src_line,tgt_line=tot_data[i]
        src_word_list = src_line.rstrip('\r\n').replace('.', '').split(' ')
        tgt_word_list = tgt_line.rstrip('\r\n').replace('.', '').split(' ')
        if len(src_word_list)>=max_len or len(tgt_word_list)>=max_len:
            continue
        
        if check(src_word_list,vocab1)==0 or check(tgt_word_list,vocab2)==0:
            continue

        if i<train_num:
            for word in src_word_list[:-1]:
                if word in vocab1:
                    fout1.write(str(vocab1[word])+' ')
                else:
                    fout1.write(str(vocab1['<unk>']) + ' ')
            if src_word_list[-1] in vocab1:
                fout1.write(str(vocab1[src_word_list[-1]]) + '\n')
            else:
                fout1.write(str(vocab1['<unk>']) + '\n')
            for word in tgt_word_list[:-1]:
                if word in vocab2:
                    fout3.write(str(vocab2[word])+' ')
                else:
                    fout3.write(str(vocab2['<unk>']) + ' ')
            if tgt_word_list[-1] in vocab2:
                fout3.write(str(vocab2[tgt_word_list[-1]]) + '\n')
            else:
                fout3.write(str(vocab2['<unk>']) + '\n')
        elif i>=train_num and i<len(src_data)-test_num:
            for word in src_word_list[:-1]:
                if word in vocab1:
                    fout2.write(str(vocab1[word])+' ')
                else:
                    fout2.write(str(vocab1['<unk>']) + ' ')
            if src_word_list[-1] in vocab1:
                fout2.write(str(vocab1[src_word_list[-1]]) + '\n')
            else:
                fout2.write(str(vocab1['<unk>']) + '\n')
            for word in tgt_word_list[:-1]:
                if word in vocab2:
                    fout4.write(str(vocab2[word])+' ')
                else:
                    fout4.write(str(vocab2['<unk>']) + ' ')
            if tgt_word_list[-1] in vocab2:
                fout4.write(str(vocab2[tgt_word_list[-1]]) + '\n')
            else:
                fout4.write(str(vocab2['<unk>']) + '\n')
        else:
            for word in src_word_list[:-1]:
                if word in vocab1:
                    fout_test_src.write(str(vocab1[word])+' ')
                else:
                    fout_test_src.write(str(vocab1['<unk>']) + ' ')
            if src_word_list[-1] in vocab1:
                fout_test_src.write(str(vocab1[src_word_list[-1]]) + '\n')
            else:
                fout_test_src.write(str(vocab1['<unk>']) + '\n')
            for word in tgt_word_list[:-1]:
                if word in vocab2:
                    fout_test_tgt.write(str(vocab2[word])+' ')
                else:
                    fout_test_tgt.write(str(vocab2['<unk>']) + ' ')
            if tgt_word_list[-1] in vocab2:
                fout_test_tgt.write(str(vocab2[tgt_word_list[-1]]) + '\n')
            else:
                fout_test_tgt.write(str(vocab2['<unk>']) + '\n')
        ind+=1
    