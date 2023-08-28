import torch
import os
from utils.dataset import loading_multimodal_data,bert_extraText, loading_AudioVision_data

import random
import numpy as np
import PIL
import torchvision.transforms as transforms

import sys
from torchvision.transforms import InterpolationMode
import yaml



NORMAL_MEAN = [0.5, 0.5, 0.5]
NORMAL_STD = [0.5, 0.5, 0.5]
SWIN_IMG_SIZE = 224


def get_multimodal_data(args, split='train'):

    print('加载目标域数据集MELD的multimodal_'+args.choice_modality+'_'+split+'...')

    print('add_or_not_emo_embed', args.add_or_not_emo_embed)

    print(f"  - Creating new {split} data")
    data = loading_multimodal_data(args,split)  
    return data

#---------------------------------------------------------------------------------------------------------------------------------------------#

def get_meld_text(args, split='train'):
    data_path = os.path.join(args.dataset_save_path, f'm3ed_unimodal_{split}_{args.choice_modality}.dt')
    print('加载数据集M3ED的unimodal_'+args.choice_modality+'_'+split+'...')
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = bert_extraText(args,split)  
        torch.save(data, data_path, pickle_protocol=4)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data
#---------------------------------------------------------------------------------------------------------------------------------------------#

def get_meld_audioORvision(args, split='train'):
    data_path = os.path.join(args.dataset_save_path, f'meld_unimodal_{split}_{args.choice_modality}.dt')
    print('加载目标域数据集M3ED的unimodal_'+args.choice_modality+'_'+split+'...')
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        if args.choice_modality == 'A':
            modality_type = 'audio'
        elif args.choice_modality == 'V':
            modality_type = 'vision'
        data = loading_AudioVision_data(modality_type, args.dataset_save_path, split)  #
        torch.save(data, data_path, pickle_protocol=4)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data


def save_Multimodal_model(model, args,curr_time):
    save_model_name = os.path.join(args.save_Model_path, 'multimodal_transformer_{}_{}.pt').format(args.choice_modality,curr_time)
    print('保存Multimodal模型:'+save_model_name)
    torch.save(model, save_model_name, pickle_protocol=4)


def load_Multimodal_model(choice_modality, save_Model_path,best_model_time):
    save_model_name = 'multimodal_transformer_{}_{}.pt'.format(choice_modality,best_model_time)
    load_path = os.path.join(save_Model_path, save_model_name)
    print("&"*100)
    print('准备最佳的Multimodal模型准备测试:'+save_model_name)
    model = torch.load(load_path)
    return model

def save_Unimodal_model(model, args,curr_time):
    save_model_name = os.path.join(args.save_Model_path, 'unimodal_model_{}_{}.pt').format(args.choice_modality,curr_time)
    print('保存Unimodal模型:'+save_model_name)
    torch.save(model, save_model_name, pickle_protocol=4)

def load_Unimodal_model(choice_modality, save_Model_path,best_model_time):
    save_model_name = os.path.join(save_Model_path, 'unimodal_model_{}_{}.pt').format(choice_modality,best_model_time)
    print("&"*100)
    print('准备最佳的Unimodal模型准备测试:'+save_model_name)
    model = torch.load(save_model_name)
    return model