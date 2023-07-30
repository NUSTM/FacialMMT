# -*- encoding: utf-8 -*-

import os
import torch
import argparse
from utils.util import *
from torch.utils.data import DataLoader,RandomSampler, SequentialSampler
from train import Lite

load_project_path = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='A Facial Expression-Aware Multimodal Multi-task Learning Framework for Emotion Recognition in Multi-party Conversations')

#---------------------------------------------------------------------------------------------------------------------------------------------#
'''MELD dataset loading'''
parser.add_argument('--load_anno_csv_path', type=str, default='/media/devin/data/meld/MELD.Raw')
parser.add_argument('--meld_text_path', type=str, default='/media/devin/data/meld/preprocess_data/ref/update')
parser.add_argument('--num_labels', type=int, default=7, help='classes number of meld') 
parser.add_argument('--data_load_path', type=str, default=os.path.join(load_project_path,'preprocess_data/'),    
                    help='path for storing the data')
parser.add_argument('--save_Model_path', default=os.path.join(load_project_path,'saved_model')) 
parser.add_argument('--plm_name', type=str, default='roberta-large', choices='[roberta-large, bert-large]')
parser.add_argument('--choice_modality', type=str, default='T+A+V', choices='[T+A+V, V]')

#---------------------------------------------------------------------------------------------------------------------------------------------#
'''Aff-Wild2 dataset loading'''
parser.add_argument('--data_folder', type=str, 
                    default='/media/devin/data/aff-wild2/cropped_aligned')
parser.add_argument('--anno_folder', type=str, 
                    default='/media/devin/data/aff-wild2/Third_ABAW_Annotations/EXPR_Classification_Challenge_used_FacialMMT/Train_Set/')
parser.add_argument('--data_list_train', type=str, 
                    default='/media/devin/data/aff-wild2/preprocess_data/FacialMMT/affwild2_train_img_path_list.txt')

'''Swin Transformer backbone loading'''
parser.add_argument("--backbone_type", type=str, default='SwinTransformer')
parser.add_argument("--backbone_conf_file", type = str, default= os.path.join(load_project_path,'modules/SwinTransformer/swin_conf.yaml'),
                    help = "The path of backbone_conf.yaml.")
parser.add_argument('--pretrained_backbone_path', type=str, 
                    default=os.path.join(load_project_path,'pretrained_model','Swin_tiny_Ms-Celeb-1M.pt'))

parser.add_argument('--tau', type=float, default=1, help='temperature parameter, default:1') 
parser.add_argument('--FacialEmoImpor_threshold', type=float, default=0.2,
                    help='filter out the emotion-blurred frames') 
#---------------------------------------------------------------------------------------------------------------------------------------------#
#tuning
parser.add_argument('--num_epochs', type=int, default=1,   
                    help='number of epochs')
parser.add_argument('--aux_lr', type=float, default=5e-5,   
                    help='initial learning rate of aux task')
parser.add_argument('--trg_lr', type=float, default=7e-6,   
                    help='initial learning rate of trg task')
parser.add_argument('--weight_decay', type=float, default=0.01, help='0.01 for FaialMMT-BERT')  
parser.add_argument('--warm_up', type=float, default=0.1, help='dynamic adjust learning rate')

parser.add_argument('--aux_batch_size', type=int, default=150, help='num of images in Aff-Wild2') 
parser.add_argument('--trg_batch_size', type=int, default=1, help='num of dialogues in MELD')    

parser.add_argument('--aux_accumulation_steps',type=int, default=1,  
                    help='gradient accumulation for src task')
parser.add_argument('--trg_accumulation_steps',type=int, default=4,  
                    help='gradient accumulation for trg task')
#-------------------------------------------
#multi-modal fusion
parser.add_argument('--crossmodal_layers_TA', type=int, default=2, help='crossmodal layers of text and audio') 
parser.add_argument('--crossmodal_num_heads_TA', type=int, default=12)
parser.add_argument('--crossmodal_attn_dropout_TA', type=float, default=0.1, help='dropout applied on the attention weights')

parser.add_argument('--crossmodal_layers_TA_V', type=int, default=2, help='crossmodal layers of text_audio and vision')
parser.add_argument('--crossmodal_num_heads_TA_V', type=int, default=12)
parser.add_argument('--crossmodal_attn_dropout_TA_V', type=float, default=0.1, help='dropout applied on the attention weights')

#---------------------------------------------------------------------------------------------------------------------------------------------#
#self-attention transformer for audio and vision
parser.add_argument('--audio_utt_Transformernum',type=int, default= 5, help='num of self-attention for audio')  
parser.add_argument('--vision_utt_Transformernum',type=int, default= 2, help='num of self-attention for vision') 

parser.add_argument('--hidden_size', type=int, default=768, help='embedding size in the transformer, 768')
parser.add_argument('--num_attention_heads', type=int, default=12, help='number of heads for the transformer network, 12')  
parser.add_argument('--intermediate_size', type=int, default=3072, help='embedding intermediate layer size, 4*hidden_size, 3072')
parser.add_argument('--hidden_act', type=str, default='gelu', help='non-linear activation function')
parser.add_argument('--hidden_dropout_prob',type=float, default=0.1, help='multimodal dropout')
parser.add_argument('--attention_probs_dropout_prob',type=float, default=0.1,help='attention dropout')
parser.add_argument('--layer_norm_eps', type=float, default=1e-12, help='1e-12')  
parser.add_argument('--initializer_range',type=int, default=0.02) 
#---------------------------------------------------------------------------------------------------------------------------------------------#

parser.add_argument('--clip', type=float, default=0.8,  
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--aux_log_interval', type=int, default=1000,
                    help='frequency of result logging') 
parser.add_argument('--trg_log_interval', type=int, default=1600,
                    help='frequency of result logging')  
parser.add_argument('--seed', type=int, default=1111, help='random seed')

#---------------------------------------------------------------------------------------------------------------------------------------------#
#Evaluate the model on the test set directly
parser.add_argument('--doEval', type=bool, default=True, help='whether to evaluate the model on the test set directly')
parser.add_argument('--load_unimodal_path', type=str, default='unimodal_model_V.pt', 
                    help='path to load the best unimodal to evaluate on the test set')
parser.add_argument('--load_multimodal_path', type=str, default= 'multimodal_model_T+A+V_RoBERTa.pt', 
                    help='path to load the best multimodal to evaluate on the test set')
parser.add_argument('--load_swin_path', type=str, default= 'best_swin_RoBERTa.pt',
                    help='path to load the best auxiliary model to evaluate on the test set')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.set_default_tensor_type('torch.FloatTensor')

if args.choice_modality == 'V':
    trg_train_data = get_meld_vision(args, 'train')  
    trg_valid_data = get_meld_vision(args, 'val')
    trg_test_data = get_meld_vision(args, 'test')
    

elif args.choice_modality == 'T+A+V':
    args.pretrainedtextmodel_path = os.path.join(load_project_path,'pretrained_model',args.plm_name)
    trg_train_data = get_multimodal_data(args, 'train')  
    trg_valid_data = get_multimodal_data(args, 'val')
    trg_test_data = get_multimodal_data(args, 'test')
    if not args.doEval:
        #auxiliary dataset
        aux_train_data = get_affwild2_dataset(args) 
        aux_train_loader = DataLoader(aux_train_data, sampler=RandomSampler(aux_train_data), batch_size=args.aux_batch_size) 
        args.aux_n_train = len(aux_train_data)

trg_train_loader = DataLoader(trg_train_data, sampler=RandomSampler(trg_train_data), batch_size=args.trg_batch_size)
trg_valid_loader = DataLoader(trg_valid_data, sampler=SequentialSampler(trg_valid_data), batch_size=args.trg_batch_size)
trg_test_loader = DataLoader(trg_test_data, sampler=SequentialSampler(trg_test_data), batch_size=args.trg_batch_size)  
args.trg_n_train, args.trg_n_valid, args.trg_n_test = len(trg_train_data), len(trg_valid_data), len(trg_test_data)   


if args.choice_modality == 'T+A+V':
    args.audio_featExtr_dim = trg_train_data.get_audio_featExtr_dim()  

if args.choice_modality in ('V', 'T+A+V'):
    args.vision_featExtr_dim = trg_train_data.get_vision_featExtr_dim()

if args.choice_modality == 'T+A+V':
    args.get_text_utt_max_lens = trg_train_data.get_text_max_utt_len()
    args.get_audio_utt_max_lens = max(trg_train_data.get_audio_max_utt_len(),trg_valid_data.get_audio_max_utt_len(),trg_test_data.get_audio_max_utt_len())

if args.choice_modality in ('V', 'T+A+V'):
    args.get_vision_utt_max_lens = max(trg_train_data.get_vision_max_utt_len(),trg_valid_data.get_vision_max_utt_len(),trg_test_data.get_vision_max_utt_len())

if __name__ == '__main__':
    print('&'*50)
    if args.doEval:
        print('Evaluating on the test set directly...')
        if args.choice_modality == 'V':
            Lite(strategy='dp', devices=1, accelerator="gpu", precision=32).run(args, None, None, None, trg_test_loader)
        elif args.choice_modality == 'T+A+V':
            Lite(strategy='dp', devices=1, accelerator="gpu", precision=16).run(args, None, None, None, trg_test_loader)
    else:
        print('Training from scratch...')
        if args.choice_modality == 'V':
            Lite(strategy='dp', devices=1, accelerator="gpu", precision=32).run(args, None, trg_train_loader, trg_valid_loader, trg_test_loader)
        elif args.choice_modality == 'T+A+V':
            Lite(strategy='dp', devices=1, accelerator="gpu", precision=16).run(args, aux_train_loader, trg_train_loader, trg_valid_loader, trg_test_loader)


