#coding:utf-8
import os.path as osp
import pandas as pd
import json
from transformers import BertTokenizer  #加载bert分词器
from collections import defaultdict


Max_seq_length = 512


def make_text_dia_utt_emo(input_json):
    labels = defaultdict(list)
    for dia_id in list(input_json.keys()):
        curr_dia = input_json[dia_id]
        curr_dia_name = dia_id
        for utt_id in list(curr_dia.keys()):
            curr_utt = curr_dia[utt_id]
            curr_utt_emo = curr_utt['label']
            labels[curr_dia_name].append(curr_utt_emo)
    return labels

# 启发式截断较长文本
def _truncate_seq_pair(tokens, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        tokens_len = []
        for i,utt in enumerate(tokens):
            tokens_len.append((i, len(utt)))
        # print("*"*10, tokens_len)
        sumlen = sum([i[1] for i in tokens_len])

        if sumlen <= max_length:   
            break
        else:
            index = sorted(tokens_len, key=lambda x:x[1], reverse=True)[0][0]
            # print(index)
            tokens[index].pop()
            # print(tokens)
    return tokens

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, sep_mask, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        # self.segment_ids = segment_ids
        self.sep_mask = sep_mask
        self.label_id = label_id

class Data_Text():
    def __init__(self, pre_output_dir, set_name, args):

        # data path  
        self.pre_output_dir = pre_output_dir  #
        self.set_name = set_name
        self.args = args

    def preprocess_data(self):

        tokenizer = BertTokenizer.from_pretrained(self.args.pretrainedtextmodel_path)

        features = []

        annot_path = osp.join(self.pre_output_dir,  self.set_name + '_utt_text_noEmo.json') 

        annot_file = json.load(open(annot_path, 'r', encoding='utf8'))

        labels = make_text_dia_utt_emo(annot_file) if self.set_name in ['train', 'val'] else None

        num_total_utt = 0

        for dia_id in list(annot_file.keys()):  
            curr_dia = annot_file[dia_id]
            temp_utt = []
            sep_mask = []
            label_id = []
            tokens = []

            for utt_id in curr_dia:
                num_total_utt += 1
                utterance = curr_dia[utt_id]['text']  
                temp_utt.append(tokenizer.tokenize(utterance))

            tokens_temp = _truncate_seq_pair(temp_utt, Max_seq_length-len(temp_utt)-1)  

            for num,tokens_utt in enumerate(tokens_temp):
                if num == 0:
                    tokens = ["[CLS]"] + tokens_utt + ["[SEP]"]
                    sep_mask = [0] * (len(tokens)-1) + [1]   
                    label_id = [0] * (len(tokens)-1) + [labels[dia_id][num]] if self.set_name in ['train', 'val'] else None
                else:
                    tokens += tokens_utt + ["[SEP]"]
                    sep_mask += [0] * len(tokens_utt) + [1]  
                    if self.set_name in ['train', 'val']:
                        label_id += [0] * len(tokens_utt) + [labels[dia_id][num]]  
                    else:
                        label_id = None

            input_ids = tokenizer.convert_tokens_to_ids(tokens) 

            input_mask = [1] * len(input_ids)  

            # Zero-pad up to the sequence length. 
            padding = [0] * (Max_seq_length - len(input_ids))  
            input_ids += padding
            input_mask += padding
            sep_mask += padding
            if self.set_name in ['train', 'val']:
                label_id += padding

            features.append(
            InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            sep_mask=sep_mask,
                            label_id=label_id))
            
        return features, num_total_utt









