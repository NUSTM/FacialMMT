#coding:utf-8
import os.path as osp
import pandas as pd
import json
from transformers import RobertaTokenizer  
from transformers import BertTokenizer 
from collections import defaultdict

Max_seq_length = 512

def make_text_dia(csv_path):

    df = pd.read_csv(csv_path, encoding='utf8')
    dia_utt_list = defaultdict(list)
    for _, row in df.iterrows():
        dia_num = int(row['Dialogue_ID'])
        utt_num = int(row['Utterance_ID'])
        dia_utt_list[str(dia_num)].append(f'dia{dia_num}_utt{utt_num}')
    return dia_utt_list


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
        # print(sumlen)
        '''
        MELD中只有test集合中的dialogue17的长度会很大, 剩下的dialogue都没有超过512
        '''
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

    def __init__(self, input_ids, input_mask, sep_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.sep_mask = sep_mask

class MELD():
    def __init__(self, load_anno_csv,pretrainedBertPath, meld_text_path, set_name):

        # data path  
        self.load_anno_csv = load_anno_csv    #
        self.pretrainedBertPath = pretrainedBertPath
        self.meld_text_path = meld_text_path
        self.set_name = set_name

    def preprocess_data(self):

        if self.pretrainedBertPath.split('/')[-1] == 'roberta-large':
            print('  - Loading RoBERTa...')
            tokenizer = RobertaTokenizer.from_pretrained(self.pretrainedBertPath)
        elif self.pretrainedBertPath.split('/')[-1] == 'bert-large':
            print('  - Loading Bert...')
            tokenizer = BertTokenizer.from_pretrained(self.pretrainedBertPath)

        features = []
        csv_path = osp.join(self.load_anno_csv,self.set_name+'_sent_emo.csv')
        int2name = make_text_dia(csv_path) #

        text_root = osp.join(self.meld_text_path, self.set_name +'_text.json')
        with open(text_root, 'r') as load_f:
            load_dict = json.load(load_f)

        for dia_id in list(int2name.keys()):   #demo [:4]

            temp_utt = []
            sep_mask = []
            tokens = []

            for utt_id in int2name[dia_id]:
                utterance = load_dict[utt_id]['txt'][0] 
                temp_utt.append(tokenizer.tokenize(utterance))
            
            if self.pretrainedBertPath.split('/')[-1] == 'roberta-large':
                tokens_temp = _truncate_seq_pair(temp_utt, Max_seq_length-34*2)  #这是计算单独的时候, 之后每个dialogue下的utterance concat一起的时候, 每个utterance会加上</s>A</s>
            elif self.pretrainedBertPath.split('/')[-1] == 'bert-large':
                tokens_temp = _truncate_seq_pair(temp_utt, Max_seq_length-34)  #这是计算单独的时候, 之后每个dialogue下的utterance concat一起的时候, 每个utterance后面要加上[sep]

            for num,tokens_utt in enumerate(tokens_temp):
                if num == 0:
                    if self.pretrainedBertPath.split('/')[-1] == 'roberta-large':
                        tokens = ["<s>"] + tokens_utt + ["</s>"]
                    elif self.pretrainedBertPath.split('/')[-1] == 'bert-large':
                        tokens = ["[CLS]"] + tokens_utt + ["[SEP]"]
                    sep_mask = [0] * (len(tokens)-1) + [1]  
                else:
                    if self.pretrainedBertPath.split('/')[-1] == 'roberta-large':
                        # <s> A </s></s> B </s>
                        tokens += ["</s>"] + tokens_utt + ["</s>"]
                        sep_mask += [0] * (len(tokens_utt)+1) + [1]  
                    elif self.pretrainedBertPath.split('/')[-1] == 'bert-large':
                        #[CLS] A [SEP] B [SEP]
                        tokens += tokens_utt + ["[SEP]"]
                        sep_mask += [0] * len(tokens_utt) + [1]  

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)  

            # Zero-pad up to the sequence length. 对不够512长度的填充0
            padding = [0] * (Max_seq_length - len(input_ids))  
            input_ids += padding
            input_mask += padding
            sep_mask += padding

            features.append(
            InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            sep_mask=sep_mask))
        return features









