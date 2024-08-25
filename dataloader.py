#import youtokentome
import codecs
import os
import torch
from random import shuffle
from itertools import groupby
from torch.nn.utils.rnn import pad_sequence
#from word2keypress import Keyboard
import string
import pickle



class PswLoader(object):

    def __init__(self, data_folder,training, batch_size,  tokens_in_batch=0):
        """
        :param data_folder: folder containing the source and target language data files
        :param source_suffix: the filename suffix for the source language files
        :param target_suffix: the filename suffix for the target language files
        :param split: train, or val, or test?
        :param tokens_in_batch: the number of target language tokens in each batch
        """
        self.tokens_in_batch = tokens_in_batch
        self.batch_size=batch_size

        # Is this for training?
        self.for_training = training


        # Load data
        if training:
            with codecs.open(os.path.join(data_folder, 'raw','src_train.txt'), "r", encoding="utf-8") as f:
                source_data = f.readlines()[:]

            with codecs.open(os.path.join(data_folder, 'raw','tgt_train.txt'), "r", encoding="utf-8") as f:
                target_data = f.readlines()[:]
        else:
            with codecs.open(os.path.join(data_folder, 'raw','src_val.txt'), "r", encoding="utf-8") as f:
                source_data = f.readlines()[:]

            with codecs.open(os.path.join(data_folder, 'raw','tgt_val.txt'), "r", encoding="utf-8") as f:
                target_data = f.readlines()[:]
        source_data=[list(x.rstrip('\r\n').split(' ')) for x in source_data]
        target_data=[list(x.rstrip('\r\n').split(' ')) for x in target_data]
        assert len(source_data) == len(target_data), "There are a different number of source or target sequences!"

        source_lengths = [len(s) for s in source_data]
        target_lengths = [len(t)+2 for t in target_data]  # target language sequences have <BOS> and <EOS> tokens
        self.data = list(zip(source_data, target_data, source_lengths, target_lengths))




        if self.for_training:
            self.data.sort(key=lambda x: x[3])
        self.total_pair=len(source_data)

        # Create batches
        self.create_batches()

        with open(os.path.join(data_folder, 'raw','src_vocab.pkl'), "rb") as f:
            src_vocab=pickle.load(f)
        src_idx2vocab={}
        for key in src_vocab:
            src_idx2vocab[src_vocab[key]]=key
        with open(os.path.join(data_folder, 'raw','tgt_vocab.pkl'), "rb") as f:
            tgt_vocab=pickle.load(f)
        tgt_idx2vocab={}
        for key in tgt_vocab:
            tgt_idx2vocab[tgt_vocab[key]]=key

        self.src_vocab_size=len(src_vocab)
        self.tgt_vocab_size = len(tgt_vocab)
        self.src_vocab=src_vocab
        self.src_idx2vocab=src_idx2vocab

        self.tgt_vocab=tgt_vocab
        self.tgt_idx2vocab=tgt_idx2vocab


    def create_batches(self):
        """
        Prepares batches for one epoch.
        """

        # If training
        self.all_batches=list()
        if self.for_training:
            i=0
            while i<len(self.data):
                self.all_batches.extend([self.data[i:i+self.batch_size]])
                i+=self.batch_size

            if i!=len:
                i-=self.batch_size
                self.all_batches.extend([self.data[i:]])
            # Shuffle batches
            shuffle(self.all_batches)
            self.n_batches = len(self.all_batches)
            self.current_batch = -1
        else:
            i=0
            while i<len(self.data):
                self.all_batches.extend([self.data[i:i+self.batch_size]])
                i+=self.batch_size
            #针对最后一个batch
            if i!=len:
                i-=self.batch_size
                self.all_batches.extend([self.data[i:]])
            # Shuffle batches
            shuffle(self.all_batches)
            self.n_batches = len(self.all_batches)
            self.current_batch = -1


    def __iter__(self):
        """
        Iterators require this method defined.
        """
        return self



    def __next__(self):
        """
        Iterators require this method defined.

        :returns: the next batch, containing:
            source language sequences, a tensor of size (N, encoder_sequence_pad_length)
            target language sequences, a tensor of size (N, decoder_sequence_pad_length)
            true source language lengths, a tensor of size (N)
            true target language lengths, typically the same as decoder_sequence_pad_length as these sequences are bucketed by length, a tensor of size (N)
        """
        # Update current batch index
        self.current_batch += 1
        try:
            source_data, target_data, source_lengths, target_lengths = zip(*self.all_batches[self.current_batch])
        # Stop iteration once all batches are iterated through
        except IndexError:
            raise StopIteration

        source_data_id=[]
        for line in source_data:
            tmp=[int(x) for x in line]
            source_data_id.append(tmp)

        target_data_id=[]
        for line in  target_data:
            tmp=[self.tgt_vocab['<s>']]
            tmp += [int(x) for x in line]
            tmp.append(self.tgt_vocab['</s>'])
            target_data_id.append(tmp)

        # Convert source and target sequences as padded tensors
        source_data = pad_sequence(sequences=[torch.LongTensor(s) for s in source_data_id],
                                   batch_first=True,
                                   padding_value=0)
        target_data = pad_sequence(sequences=[torch.LongTensor(t) for t in target_data_id],
                                   batch_first=True,
                                   padding_value=0)

        # Convert lengths to tensors
        source_lengths = torch.LongTensor(source_lengths)
        target_lengths = torch.LongTensor(target_lengths)

        return source_data, target_data, source_lengths, target_lengths


class TestPswLoader(object):

    def __init__(self, data_folder,training, batch_size,  tokens_in_batch=0):
        """
        :param data_folder: folder containing the source and target language data files
        :param source_suffix: the filename suffix for the source language files
        :param target_suffix: the filename suffix for the target language files
        :param split: train, or val, or test?
        :param tokens_in_batch: the number of target language tokens in each batch
        """
        self.tokens_in_batch = tokens_in_batch
        self.batch_size=batch_size

        # Is this for training?
        self.for_training = training



        with codecs.open(os.path.join(data_folder, 'raw','src_test.txt'), "r", encoding="utf-8") as f:
            source_data = f.readlines()
        with codecs.open(os.path.join(data_folder, 'raw','tgt_test.txt'), "r", encoding="utf-8") as f:
            target_data = f.readlines()
        print(len(source_data))
        source_data=[list(x.rstrip('\r\n').split(' ')) for x in source_data]
        target_data =[list(x.rstrip('\r\n').split(' ')) for x in target_data]

        source_lengths = [len(s) for s in source_data]#获得每个元素进行bpe划分之后的长度

        self.data = list(zip(source_data, source_lengths,target_data))



        # Create batches
        self.create_batches()

        #
        with open(os.path.join(data_folder, 'raw','src_vocab.pkl'), "rb") as f:
            src_vocab=pickle.load(f)
        src_idx2vocab={}
        for key in src_vocab:
            src_idx2vocab[src_vocab[key]]=key
        with open(os.path.join(data_folder, 'raw','tgt_vocab.pkl'), "rb") as f:
            tgt_vocab=pickle.load(f)
        tgt_idx2vocab={}
        for key in tgt_vocab:
            tgt_idx2vocab[tgt_vocab[key]]=key

        self.src_vocab_size=len(src_vocab)
        self.tgt_vocab_size = len(tgt_vocab)
        self.src_vocab=src_vocab
        self.src_idx2vocab=src_idx2vocab

        self.tgt_vocab=tgt_vocab
        self.tgt_idx2vocab=tgt_idx2vocab



    def create_batches(self):
        # Simply return once pair at a time
        self.all_batches = [[d] for d in self.data]
        self.n_batches = len(self.all_batches)
        #print("all batches:%d"%self.n_batches)
        self.current_batch = -1

    def __iter__(self):
        """
        Iterators require this method defined.
        """
        return self



    def __next__(self):

        # Update current batch index
        self.current_batch += 1
        try:
            source_data, source_lengths,target_data = zip(*self.all_batches[self.current_batch])
        # Stop iteration once all batches are iterated through
        except IndexError:
            raise StopIteration

        source_data_id=[]
        for line in source_data:
            tmp = [int(x) for x in line]
            source_data_id.append(tmp)


        # Convert source and target sequences as padded tensors
        source_data = pad_sequence(sequences=[torch.LongTensor(s) for s in source_data_id],
                                   batch_first=True,
                                   padding_value=0)


        # Convert lengths to tensors
        source_lengths = torch.LongTensor(source_lengths)


        return source_data, source_lengths,target_data
