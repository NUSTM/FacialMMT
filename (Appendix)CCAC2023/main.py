import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch

TORCH_DISTRIBUTED_DEBUG = 'INFO'

import argparse

from utils.util import *
from torch.utils.data import DataLoader,RandomSampler, SequentialSampler
from train import Lite
# import json
load_project_path = os.path.abspath(os.path.dirname(__file__))

load_dataset_path = os.path.join(load_project_path, 'data')

parser = argparse.ArgumentParser(description='Facial-Aware MERC for M3ED dataset')

parser.add_argument('--m3ed_project_path', type=str, default=load_dataset_path)
parser.add_argument('--pretrainedtextmodel_path', type=str, default= os.path.join(load_dataset_path, 'public_pretrained_model/chinese-roberta-wwm-ext-large'),
                    help='the type of pretrained model path')  
#数据方面
parser.add_argument('--num_labels', type=int, default=7, help='classes number of m3ed') 

parser.add_argument('--save_Model_path', default= os.path.join(load_project_path, 'save'))
parser.add_argument('--choice_modality', type=str, default='T+A+V')

#---------------------------------------------------------------------------------------------------------------------------------------------#
#是否执行pipeline形式的帧级别情绪指导(为目标数据的视觉表示增加7分类的情绪分布)
parser.add_argument('--add_or_not_emo_embed', type=bool, default=True, help='True or False')  

#---------------------------------------------------------------------------------------------------------------------------------------------#
#tuning
parser.add_argument('--num_epochs', type=int, default=1,   #迭代次数
                    help='number of epochs')
parser.add_argument('--trg_lr', type=float, default=7e-5,   #多模态模型上的学习率
                    help='initial learning rate (default: )')
parser.add_argument('--trg_batch_size', type=int, default=4, help='num of utterance/dialogue in M3ED')   #m3ed中支持的utterance个数或者dialgoue个数
parser.add_argument('--trg_accumulation_steps',type=int, default=1,  
                    help='gradient accumulation for trg task')
#-------------------------------------------
#多模态融合方式
parser.add_argument('--modalityFuse', type=str, default='crossmodal', help='choice of modality fusing[concat, crossmodal]')
parser.add_argument('--crossmodal_layers_TA', type=int, default=2, help='crossmodal layers of text and audio') 
parser.add_argument('--crossmodal_num_heads_TA', type=int, default=12)
parser.add_argument('--crossmodal_attn_dropout_TA', type=float, default=0.1, help='dropout applied on the attention weights')

parser.add_argument('--crossmodal_layers_TV', type=int, default=2, help='crossmodal layers of text and vision') 
parser.add_argument('--crossmodal_num_heads_TV', type=int, default=12)
parser.add_argument('--crossmodal_attn_dropout_TV', type=float, default=0.1, help='dropout applied on the attention weights')

parser.add_argument('--crossmodal_layers_TA_V', type=int, default=2, help='crossmodal layers of text_audio and vision')
parser.add_argument('--crossmodal_num_heads_TA_V', type=int, default=12)
parser.add_argument('--crossmodal_attn_dropout_TA_V', type=float, default=0.1, help='dropout applied on the attention weights')

parser.add_argument('--crossmodal_gelu_dropout', type=float, default=0.0, help='dropout applied on the first layer of the residual block')
parser.add_argument('--crossmodal_res_dropout', type=float, default=0.0, help='dropout applied on the residual block')
parser.add_argument('--crossmodal_embed_dropout', type=float, default=0.1, help='')
parser.add_argument('--crossmodal_attn_mask', type=bool, default=False, help='whether to apply mask on the attention weights')

#---------------------------------------------------------------------------------------------------------------------------------------------#
parser.add_argument('--warm_up', type=float, default=0.1, help='dynamic adjust learning rate')

#语音、视觉  self-attention transformer 
parser.add_argument('--audio_utt_Transformernum',type=int, default= 5, help='num of self-attention for audio')  
parser.add_argument('--vision_utt_Transformernum',type=int, default= 2, help='num of self-attention for vision') 

parser.add_argument('--hidden_size', type=int, default=768, help='embedding size in the transformer, 768')
parser.add_argument('--num_attention_heads', type=int, default=12, help='number of heads for the transformer network, 12')  
parser.add_argument('--intermediate_size', type=int, default=3072, help='embedding intermediate layer size, 4*hidden_size, 3072')

parser.add_argument('--hidden_act', type=str, default='gelu', help='non-linear activation function')
parser.add_argument('--hidden_dropout_prob',type=float, default=0.1, help='transformer hidden dropout')
parser.add_argument('--attention_probs_dropout_prob',type=float, default=0.1,help='attention dropout')
parser.add_argument('--layer_norm_eps', type=float, default=1e-12, help='1e-12')  
parser.add_argument('--initializer_range',type=int, default=0.02) 
#---------------------------------------------------------------------------------------------------------------------------------------------#

parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')  #权重衰减项
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')  #early stopping 
parser.add_argument('--clip', type=float, default=0.8,  
                    help='gradient clip value (default: 0.8)')  
parser.add_argument('--trg_log_interval', type=int, default=5,
                    help='frequency of result logging (default: 50)')  #多少个trg_batch打印一个结果
parser.add_argument('--seed', type=int, default=1111,     
                    help='random seed')

parser.add_argument('--uttORdia', type=str, default='dia', choices=['utt','dia']) #加载utt级别模型还是dia级别模型
parser.add_argument('--conduct_emo_eval', type=bool, default=True)  #是否直接进行情感识别的评估
parser.add_argument('--load_best_model_path', type=str, default=os.path.join(load_dataset_path,'multimodal_transformer_T+A+V_06-10-01-51-42.pt'))  #选择的模态 

args = parser.parse_args()
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.set_default_tensor_type('torch.FloatTensor')


if args.conduct_emo_eval:
    trg_test_data = get_multimodal_data(args, 'test') 
    trg_test_loader = DataLoader(trg_test_data, sampler=SequentialSampler(trg_test_data), batch_size=args.trg_batch_size) 
    args.trg_n_test = len(trg_test_data) 
else:
    trg_train_data = get_multimodal_data(args, 'train')  
    trg_valid_data = get_multimodal_data(args, 'val')
    trg_train_loader = DataLoader(trg_train_data, sampler=RandomSampler(trg_train_data), batch_size=args.trg_batch_size)
    trg_valid_loader = DataLoader(trg_valid_data, sampler=SequentialSampler(trg_valid_data), batch_size=args.trg_batch_size)
    args.trg_n_train, args.trg_n_valid = len(trg_train_data), len(trg_valid_data)


args.audio_featExtr_dim = trg_train_data.get_audio_featExtr_dim() if not args.conduct_emo_eval else trg_test_data.get_audio_featExtr_dim()

args.vision_featExtr_dim = trg_train_data.get_vision_featExtr_dim() if not args.conduct_emo_eval else trg_test_data.get_vision_featExtr_dim()


if args.conduct_emo_eval: 
    args.get_text_utt_max_lens = trg_test_data.get_text_max_utt_len()  
    args.get_audio_utt_max_lens = trg_test_data.get_audio_max_utt_len()
    args.get_vision_utt_max_lens = trg_test_data.get_vision_max_utt_len()
else:
    args.get_text_utt_max_lens = trg_train_data.get_text_max_utt_len()
    args.get_audio_utt_max_lens = max(trg_train_data.get_audio_max_utt_len(),trg_valid_data.get_audio_max_utt_len())
    args.get_vision_utt_max_lens = max(trg_train_data.get_vision_max_utt_len(),trg_valid_data.get_vision_max_utt_len())

if __name__ == '__main__':

    if not args.conduct_emo_eval:
    #打印所有参数
        print('args:', args)
        print('&'*100)
        if args.add_or_not_emo_embed:
            print('即将执行{}级别的pipeline形式的多模态模型训练...'.format(args.uttORdia))
        Lite(strategy='dp', devices=1, accelerator="gpu", precision=32).run(args, trg_train_loader, trg_valid_loader, None)
    else:
        #直接在测试集trg_test_loader上测试
        print('&'*100)
        print('直接在测试集上出{}级别模型的预测结果, 供主办方进行检测...'.format(args.uttORdia))
        Lite(strategy='dp', devices=1, accelerator="gpu", precision=32).run(args, None, None, trg_test_loader)

