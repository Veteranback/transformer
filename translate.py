import torch
import torch.nn.functional as F
#import youtokentome
import math
#from word2keypress import Keyboard
from dataloader import TestPswLoader,PswLoader
import argparse
import time
from nltk.translate.bleu_score import  sentence_bleu
from rouge import Rouge
from collections import defaultdict
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--data',type=str,default='WMT14')
parser.add_argument('--size', type=int, default=10)
parser.add_argument('--beam', type=int, default=5)
parser.add_argument('--checkpoint', type=str, default='model_WMT14/transformer.pth.tar')
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()
device = args.device



# Transformer model
checkpoint = torch.load(args.checkpoint)
model = checkpoint['model'].to(device)

# 设置device
model.encoder.device = args.device
model.decoder.device = args.device
model.encoder.device = args.device
model.eval()
#kb=Keyboard()

def translate(source_sequence,vocab_size,id2char,char2id, beam_size=4, length_norm_coefficient=0.6):
    global args
    """
    Translates a source language sequence to the target language, with beam search decoding.

    :param source_sequence: the source language sequence, either a string or tensor of bpe-indices
    :param beam_size: beam size
    :param length_norm_coefficient: co-efficient for normalizing decoded sequences' scores by their lengths
    :return: the best hypothesis, and all candidate hypotheses
    """
    with torch.no_grad():
        # Beam size
        k = beam_size

        # Minimum number of hypotheses to complete
        n_completed_hypotheses = args.size





        encoder_sequences = source_sequence.to(device)  # (1, source_sequence_length)
        encoder_sequence_lengths = torch.LongTensor([encoder_sequences.size(1)]).to(device)  # (1)

        # Encode
        encoder_sequences = model.encoder(encoder_sequences=encoder_sequences,
                                          encoder_sequence_lengths=encoder_sequence_lengths)  # (1, source_sequence_length, d_model)

        # Our hypothesis to begin with is just <BOS>
        hypotheses = torch.LongTensor([[char2id['<s>']]]).to(device)  # (1, 1)
        hypotheses_lengths = torch.LongTensor([hypotheses.size(1)]).to(device)  # (1)

        # Tensor to store hypotheses' scores; now it's just 0
        hypotheses_scores = torch.zeros(1).to(device)  # (1)

        # Lists to store completed hypotheses and their scores
        completed_hypotheses = list()
        completed_hypotheses_scores = list()

        # Start decoding
        step = 1

        # Assume "s" is the number of incomplete hypotheses currently in the bag; a number less than or equal to "k"
        # At this point, s is 1, because we only have 1 hypothesis to work with, i.e. "<BOS>"
        while True:
            s = hypotheses.size(0)
            if s==0:
                break
            decoder_sequences = model.decoder(decoder_sequences=hypotheses,
                                              decoder_sequence_lengths=hypotheses_lengths,
                                              encoder_sequences=encoder_sequences.repeat(s, 1, 1),
                                              encoder_sequence_lengths=encoder_sequence_lengths.repeat(
                                                  s))  # (s, step, vocab_size)

            # Scores at this step
            scores = decoder_sequences[:, -1, :]  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=-1)  # (s, vocab_size)

            # Add hypotheses' scores from last step to scores at this step to get scores for all possible new hypotheses
            scores = hypotheses_scores.unsqueeze(1) + scores  # (s, vocab_size)
            norm = math.pow(((5 + step) / (5 + 1)), length_norm_coefficient)
            # for i in range(scores.shape[0]):
            #     end_ID=char2id[chr(2)]
            #     if len(hypotheses.tolist()[i])-1>3:#长度限制
            #         completed_hypotheses.append(hypotheses.tolist()[i])
            #         completed_hypotheses_scores.append(scores[i,end_ID].item()/norm)
            #     if args.device=='cpu':
            #         scores[i,end_ID]=-10000000
            #     else:
            #         scores.cpu()[i,end_ID]=-10000000




            # Unroll and find top k scores, and their unrolled indices
            next_beam=min(scores.view(-1).shape[0],500)
            top_k_hypotheses_scores, unrolled_indices = scores.view(-1).topk(next_beam, 0, True, True)  # (k)

            # Convert unrolled indices to actual indices of the scores tensor which yielded the best scores
            prev_word_indices = unrolled_indices // vocab_size  # (k)
            next_word_indices = unrolled_indices % vocab_size  # (k)

            # Construct the the new top k hypotheses from these indices
            top_k_hypotheses = torch.cat([hypotheses[prev_word_indices], next_word_indices.unsqueeze(1)],
                                         dim=1)  # (k, step + 1)

            # Which of these new hypotheses are complete (reached <EOS>)?
            complete = next_word_indices ==char2id['</s>']  # (k), bool

            # Set aside completed hypotheses and their scores normalized by their lengths
            # For the length normalization formula, see
            # "Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation"
            if step > 3:
                completed_hypotheses.extend(top_k_hypotheses[complete].tolist())

                completed_hypotheses_scores.extend((top_k_hypotheses_scores[complete] / norm).tolist())

            if len(completed_hypotheses)>n_completed_hypotheses:
                break


            # Else, continue with incomplete hypotheses
            hypotheses = top_k_hypotheses[~complete]  # (s, step + 1)
            hypotheses_scores = top_k_hypotheses_scores[~complete]  # (s)
            hypotheses_lengths = torch.LongTensor(hypotheses.size(0) * [hypotheses.size(1)]).to(device)  # (s)

            # Stop if things have been going on for too long
            if step > 30:
                break
            step += 1

        # If there is not a single completed hypothesis, use partial hypotheses
        # if len(completed_hypotheses) == 0:
        #     completed_hypotheses = hypotheses.tolist()
        #     completed_hypotheses_scores = hypotheses_scores.tolist()

        # Decode the hypotheses
        all_hypotheses = list()
        i=0
        for pred_psw_id_list in completed_hypotheses:
            scores=completed_hypotheses_scores[i]
            # re=''
            # for char in pred_psw_id_list:
            #     if char in id2char and id2char[char] not in [chr(1),chr(2),0]:
            #         re+=id2char[char]
            #     elif char==0 or id2char[char]=='</s>':#遇到结束字符或者是填充字符
            #         break

            all_hypotheses.append([pred_psw_id_list,scores])
            i+=1

        #然后按照scores降序排序
        all_hypotheses=sorted(all_hypotheses,key=lambda x:x[1],reverse=True)


        return  all_hypotheses



def bleu(src,tgt):
    
    src=[str(x) for x in src]
    score1 = sentence_bleu([src], tgt,weights=(1,0,0,0))
    score2 = sentence_bleu([src], tgt,weights=(0,1,0,0))
    score3 = sentence_bleu([src], tgt,weights=(0,0,1,0))
    score4 = sentence_bleu([src], tgt,weights=(0,0,0,1))

    return score1,score2,score3,score4

def rouge_score(src,tgt):
    rouge = Rouge()
    src=' '.join([str(x) for x in src])
    tgt=' '.join(tgt)
    
    rouge_re = rouge.get_scores(hyps=src, refs=tgt)
    # print(rouge_re[0]["rouge-1"])
    # print(rouge_re[0]["rouge-2"]) #ROUGE-2
    # print(rouge_re[0]["rouge-l"]) #ROUGE-L
    return rouge_re

if __name__ == '__main__':
    val_loader = TestPswLoader(data_folder=args.data,
                           training=0,
                           batch_size=1)
    val_loader.create_batches()
    #fout=open(args.data+'/translate_result.txt','w')
    fout2=open(args.data+'/generate.txt','w')
    
    time_start=time.time()
    tot,bleu_socre,rouge_1,rouge_2,rouge_l=0,0,defaultdict(float),defaultdict(float),defaultdict(float)
    bleu_socre2,bleu_socre3,bleu_socre4=0,0,0
    for i, (source_sequence, source_sequence_length,target_data) in enumerate(val_loader):
        guesses=translate(source_sequence,val_loader.tgt_vocab_size+1,val_loader.tgt_idx2vocab,val_loader.tgt_vocab,
                      beam_size=args.beam)
        if len(guesses)<1:
            pred_str=[]
        else:
            pred,score= guesses[0]
            pred=pred[1:-1]
            pred_str=[val_loader.tgt_idx2vocab[x] for x in pred]
        target_data=[val_loader.tgt_idx2vocab[int(x)] for x in target_data[0]]
        fout2.write('%s\n%s\n'%(str(pred_str),str(target_data)))
        print('------------%d -----------'%(i))
        print('pred:%s \ntgt:%s'%(str(pred_str),str(target_data)))

    #     if len(guesses)<1:
    #         bleu_re=[0,0,0,0]
    #         rouge={'rouge-1':{'f':0,'p':0,'r':0},
    #         'rouge-2':{'f':0,'p':0,'r':0},'rouge-l':{'f':0,'p':0,'r':0}}
    #     else:
    #         pred,score= guesses[0]
    #         pred=pred[1:-1]
    #         pred_str=[val_loader.tgt_idx2vocab[x] for x in pred]
    #         target_data=[val_loader.tgt_idx2vocab[int(x)] for x in target_data[0]]
    #         fout2.write('------------%d -----------\n'%(i))
    #         fout2.write('pred:%s \ntgt:%s\n'%(str(pred_str),str(target_data)))
    #         print('------------%d -----------'%(i))
    #         print('pred:%s \ntgt:%s'%(str(pred_str),str(target_data)))
    #         bleu_re=bleu(pred_str,target_data)
    #         rouge=rouge_score(pred_str,target_data)[0]

    #     print('BLEU:%s %s %s %s'%(str(bleu_re[0]),str(bleu_re[1]),str(bleu_re[2]),str(bleu_re[3])))
    #     print('ROUGE-1 r:%s p:%s f:%s'%(rouge['rouge-1']['r'],rouge['rouge-1']['p'],rouge['rouge-1']['f']))
    #     print('ROUGE-2 r:%s p:%s f:%s'%(rouge['rouge-2']['r'],rouge['rouge-2']['p'],rouge['rouge-2']['f']))
    #     print('ROUGE-l r:%s p:%s f:%s\n\n'%(rouge['rouge-l']['r'],rouge['rouge-l']['p'],rouge['rouge-l']['f']))
        
    #     fout2.write('BLEU:%s %s %s %s\n'%(str(bleu_re[0]),str(bleu_re[1]),str(bleu_re[2]),str(bleu_re[3])))
    #     fout2.write('ROUGE-1 r:%s p:%s f:%s\n'%(rouge['rouge-1']['r'],rouge['rouge-1']['p'],rouge['rouge-1']['f']))
    #     fout2.write('ROUGE-2 r:%s p:%s f:%s\n'%(rouge['rouge-2']['r'],rouge['rouge-2']['p'],rouge['rouge-2']['f']))
    #     fout2.write('ROUGE-l r:%s p:%s f:%s\n\n'%(rouge['rouge-l']['r'],rouge['rouge-l']['p'],rouge['rouge-l']['f']))
        
    #     bleu_socre+=bleu_re[0]
    #     bleu_socre2+=bleu_re[1]
    #     bleu_socre3+=bleu_re[2]
    #     bleu_socre4+=bleu_re[3]

    #     for key in rouge['rouge-1']:
    #         rouge_1[key]+=rouge['rouge-1'][key]
    #     for key in rouge['rouge-2']:
    #         rouge_2[key]+=rouge['rouge-2'][key]
    #     for key in rouge['rouge-l']:
    #         rouge_l[key]+=rouge['rouge-l'][key]
    #     tot+=1


    # fout.write('tot:%d BLEU:%s %s %s %s\n'%(tot,str(bleu_socre/tot),str(bleu_socre2/tot),str(bleu_socre3/tot),str(bleu_socre4/tot)))
    # fout.write('ROUGE-1 r:%s p:%s f:%s \n'%(rouge_1['r']/tot,rouge_1['p']/tot,rouge_1['f']/tot))
    # fout.write('ROUGE-2 r:%s p:%s f:%s \n'%(rouge_2['r']/tot,rouge_2['p']/tot,rouge_2['f']/tot))
    # fout.write('ROUGE-l r:%s p:%s f:%s \n'%(rouge_l['r']/tot,rouge_l['p']/tot,rouge_l['f']/tot))


