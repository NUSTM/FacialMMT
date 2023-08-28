import json
import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
import torch
import torchvision.transforms as transforms

# sys.path.append(os.path.join(os.getcwd(), "../.."))  #上级目录
# from util import from_image_to_embedding_no_IncepRes

from src.data_bert_extraText import Data_Text

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')



NORMAL_MEAN = [0.5, 0.5, 0.5]
NORMAL_STD = [0.5, 0.5, 0.5]
SWIN_IMG_SIZE = 224

TEXT_MAX_UTT_LEN = 35 

# MAX_DIA_IMG_LEN = 2350


import cv2
from PIL import Image
# import glob


#----------------------------------------------------------------------------------------------------#

transforms_train = transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=NORMAL_MEAN,
                                        std=NORMAL_STD) 
                                    ])

transforms_val = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=NORMAL_MEAN,
                                        std=NORMAL_STD) 
                                    ])


def from_image_to_embedding_no_IncepRes(images_path_list, set_name):

    images_lens = len(images_path_list)
    X = torch.zeros([images_lens,3,SWIN_IMG_SIZE,SWIN_IMG_SIZE])
    for i in range(images_lens):  #
        im = cv2.imread(images_path_list[i])
        '''
        修改进来是帧级别图片的size
        '''
        if im.shape[0] > SWIN_IMG_SIZE:
            im = cv2.resize(im, dsize=(SWIN_IMG_SIZE,SWIN_IMG_SIZE), interpolation=cv2.INTER_AREA)  #缩小
        if im.shape[0] < SWIN_IMG_SIZE:
            im = cv2.resize(im, dsize=(SWIN_IMG_SIZE,SWIN_IMG_SIZE), interpolation=cv2.INTER_CUBIC)  #放大

        im = Image.fromarray(im,mode='RGB')  #
        
        #数据增强
        if set_name == 'train':
            x = transforms_train(im)
        else:
            x = transforms_val(im)
        
        X[i,:] = x
    res = X.cpu().detach().numpy()
    return res


'''加载视觉或者语音单模态'''
class loading_AudioVision_data(Dataset):
    def __init__(self, modality_type, dataset_path, split_type):
        super(loading_AudioVision_data, self).__init__()
        self.modality_type = modality_type

        dataset_path = os.path.join(dataset_path, 'meld_'+split_type+'_'+modality_type+'_utt.pkl')
        dataset = pickle.load(open(dataset_path, 'rb'))
        self.modality_feature = torch.tensor(dataset[split_type][self.modality_type].astype(np.float32)).cpu().detach()  #(utt_num, utt_max_lens, extraFeature_dim)
        self.labels = torch.tensor(np.array(dataset[split_type]['labels'])).cpu().detach()   #(utt_num)
        self.utterance_mask = torch.tensor(dataset[split_type][self.modality_type+'_utt_mask']).cpu().detach() #(utt_num, utt_max_lens) 有效的置1, 无效的置0

    def get_vision_max_utt_len(self):
        return self.modality_feature.shape[1]
    
    def get_vision_featExtr_dim(self):  
        return self.modality_feature.shape[-1]    #extraFeature_dim

    def get_audio_max_utt_len(self):
        return self.modality_feature.shape[1]
    
    def get_audio_featExtr_dim(self):  
        return self.modality_feature.shape[-1]    #extraFeature_dim
    
    def __len__(self):
        return self.modality_feature.shape[0]  #返回当前集合的utterance的数量

    def __getitem__(self, index):    
        X = self.modality_feature[index]    #shape为(utt_max_lens, Feature_extra_dim)  若为进来初始化的特征，(utt_max_lens, 3, 224, 224)
        utt_mask = self.utterance_mask[index] 
        Y = self.labels[index] #(1)
        return X, utt_mask, Y.cuda() 


#-------------------------------------------------------------------------------------------------------------------------------------------------------
'''加载单文本模态'''
class loading_unimodal_text(Dataset):
    def __init__(self, features, num_utt, args, split_type):
        super(loading_unimodal_text, self).__init__()
        self.features = features

        self.text_input_ids = torch.tensor([f.input_ids for f in self.features], dtype=torch.long) #(dia_num, dia_max_len)
        self.text_input_mask = torch.tensor([f.input_mask for f in self.features], dtype=torch.long) #(dia_num, dia_max_len)
        self.text_sep_mask = torch.tensor([f.sep_mask for f in self.features], dtype=torch.long) #(dia_num, dia_max_len)
        self.text_label_ids = torch.tensor([f.label_id for f in self.features], dtype=torch.long) #(dia_num, dia_max_len)
        self.num_utt = num_utt

        utt_profile_path = os.path.join(args.m3ed_project_path, split_type + '_utt_profile.json')
        with open(utt_profile_path,'r') as rd:
            self.utt_profile = json.load(rd)

    def __len__(self):
        return self.num_utt      

    def get_text_max_utt_len(self):
        return TEXT_MAX_UTT_LEN 

    def __getitem__(self, index):     
        
        curr_utt_profile = self.utt_profile[str(index)]  
        curr_utt_name, currUtt_belg_dia_name, currUtt_belg_dia_idx, currUtt_belg_dia_len, currUtt_in_dia_idx = curr_utt_profile
        # print('正在加载'+curr_utt_name+'...')
        batch_input_ids  = self.text_input_ids[currUtt_belg_dia_idx]
        batch_input_mask = self.text_input_mask[currUtt_belg_dia_idx]
        batch_sep_mask = self.text_sep_mask[currUtt_belg_dia_idx]
        batch_label_ids = self.text_label_ids[currUtt_belg_dia_idx] #记录某个dialogue下所有Utt的标签
        
        #根据batch_sep_mask确定当前utt的真实标签位置

        uttlabelInDia_idx = []
        for k,v in enumerate(list(batch_sep_mask)):
            if v == 1:
                uttlabelInDia_idx.append(k)  
        label_idx = uttlabelInDia_idx[currUtt_in_dia_idx]  #拿到该utt在当前dialogue中的真实标签位置
        batch_label_ids = batch_label_ids[label_idx]

        return batch_input_ids, batch_input_mask, batch_sep_mask, batch_label_ids.cuda(), currUtt_in_dia_idx


def bert_extraText(args, split=None):
    pre_output_dir = args.m3ed_project_path  
    m3ed = Data_Text(pre_output_dir, split, args)
    m3ed_text_features,num_utt = m3ed.preprocess_data()  

    final_data = loading_unimodal_text(m3ed_text_features, num_utt, args, split)
    return final_data

#-------------------------------------------------------------------------------------------------------------------------------------------------------
'''加载多模态数据'''
class loading_multimodal_dataset(Dataset):
    def __init__(self, text_inputs, args, split_type):
        super(loading_multimodal_dataset, self).__init__()

        self.args = args
        self.choice_modality = args.choice_modality
        self.split_type = split_type
        self.add_or_not_emo_embed = args.add_or_not_emo_embed

        '''文本模态(整个dialogue下所有utterance串起来)输入'''
        self.text_input_ids = torch.tensor([f.input_ids for f in text_inputs], dtype=torch.long).cpu().detach() #(dia_num, max_sequence_len)
        self.text_input_mask = torch.tensor([f.input_mask for f in text_inputs], dtype=torch.long).cpu().detach() #(dia_num, max_sequence_len)
        self.text_sep_mask = torch.tensor([f.sep_mask for f in text_inputs], dtype=torch.long).cpu().detach() #(dia_num, max_sequence_len) 

        '''语音模态输入'''
        audio_path = os.path.join(args.m3ed_project_path, 'm3ed_'+split_type + '_audio_{}.pkl'.format(args.uttORdia))
        openfile_a = open(audio_path, 'rb')
        audio_data = pickle.load(openfile_a)
        self.audio_feature = audio_data[split_type]['audio']  #(num_utt, max_audio_utt_len, extraFeature_dim) or (num_dia, max_audio_dia_len, max_audio_utt_len, extraFeature_dim)
        self.audio_utterance_mask = audio_data[split_type]['audio_utt_mask']
        
        self.audio_dialogue_mask = audio_data[split_type]['audio_dia_mask'] if args.uttORdia == 'dia' else None  
        
        if args.uttORdia == 'utt':
            self.audio_max_utt_len, self.audio_feat_dim = self.audio_feature.shape[1], self.audio_feature.shape[-1]
        else:
            self.audio_max_dia_len, self.audio_max_utt_len, self.audio_feat_dim = self.audio_feature.shape[1], self.audio_feature.shape[2], self.audio_feature.shape[-1]
        
        '''三模态统一标签信息'''
        if split_type in ['train', 'val']:
            self.labels = torch.tensor(audio_data[split_type]['labels'],dtype=torch.long).cpu().detach()   #(num_dia, )  
        openfile_a.close()
        
        if args.uttORdia == 'utt':
            utt_profile_path = os.path.join(args.m3ed_project_path, split_type + '_utt_profile.json')
            with open(utt_profile_path,'r') as rd:
                self.utt_profile = json.load(rd)
        else:
            dia_profile_path = os.path.join(args.m3ed_project_path, split_type + '_num_utt_in_dia.json')
            with open(dia_profile_path,'r') as rd:
                self.dia_profile = json.load(rd)

        '''视觉模态输入'''
        print('  - 加载人脸序列...')

        vision_path = os.path.join(args.m3ed_project_path, 'm3ed_' + split_type + '_vision_{}.pkl'.format(args.uttORdia))
        openfile_v = open(vision_path, 'rb')
        vision_data = pickle.load(openfile_v)

        self.vision_feature = vision_data[split_type]['vision'] #(num_utt, max_audio_utt_len, extraFeature_dim) or (num_dia, max_audio_dia_len, max_audio_utt_len, extraFeature_dim)
        self.vision_utterance_mask = vision_data[split_type]['vision_utt_mask'] 

        self.vision_dialogue_mask = vision_data[split_type]['vision_dia_mask'] if args.uttORdia == 'dia' else None

        openfile_v.close()
        # openfile_v_2.close()
        if args.uttORdia == 'utt':
            self.vision_max_utt_len, self.vision_feat_dim = self.vision_feature.shape[1], self.vision_feature.shape[-1] 
        else:
            self.vision_max_dia_len, self.vision_max_utt_len, self.vision_feat_dim = self.vision_feature.shape[1], self.vision_feature.shape[2], self.vision_feature.shape[-1]


    def __len__(self):

        return self.audio_feature.shape[0]  #返回该集合下utterance的个数或者dialogue的个数

    def get_text_max_utt_len(self):
        return TEXT_MAX_UTT_LEN 

    def get_audio_max_utt_len(self):
        return self.audio_max_utt_len #57

    def get_vision_max_utt_len(self):
        return self.vision_max_utt_len  #80

    def get_audio_featExtr_dim(self):  #通过wav2vec2.0获得的768维
        return self.audio_feat_dim
    
    def get_vision_featExtr_dim(self):  #通过InceptionResnetv1获得的512维
        return self.vision_feat_dim

    def __getitem__(self, index):    #随机采样第index个的utterance或者diaalogue

        if self.args.uttORdia == 'utt':
            curr_utt_profile = self.utt_profile[str(index)]  
            curr_utt_name, currUtt_belg_dia_name, currUtt_belg_dia_idx, currUtt_belg_dia_len, currUtt_in_dia_idx = curr_utt_profile
            # print('curr_utt_name: ', curr_utt_name)
            '''文本模态'''
            curr_text_input_ids  = self.text_input_ids[currUtt_belg_dia_idx]
            curr_text_input_mask = self.text_input_mask[currUtt_belg_dia_idx]
            curr_text_sep_mask = self.text_sep_mask[currUtt_belg_dia_idx]
        elif self.args.uttORdia == 'dia':
            curr_numUtt_in_dia = self.dia_profile[str(index)]
            curr_text_input_ids  = self.text_input_ids[index]
            curr_text_input_mask = self.text_input_mask[index]
            curr_text_sep_mask = self.text_sep_mask[index]

        '''标签信息'''
        if self.split_type in ['train','val']:
            curr_label_ids = self.labels[index] 

        '''语音模态'''
        audio_inputs = self.audio_feature[index] #加载当前utterance     .shape为(utt_max_lens, Feature_extra_dim) 
        audio_mask = self.audio_utterance_mask[index]  #(utt_max_lens)

        '''视觉模态'''
        vision_inputs = self.vision_feature[index] #加载当前utterance     .shape为(utt_max_lens, Feature_extra_dim) 
        vision_mask = self.vision_utterance_mask[index]  #(utt_max_lens)
        curr_utt_embed = vision_inputs
        
        if self.args.uttORdia == 'dia':
            dia_mask = self.vision_dialogue_mask[index]  

        if self.args.uttORdia == 'utt':
            if self.split_type in ['train','val']:
                return curr_text_input_ids, curr_text_input_mask, curr_text_sep_mask, audio_inputs, audio_mask, \
                    curr_utt_embed,vision_mask, curr_label_ids, currUtt_in_dia_idx
            else:
                return curr_text_input_ids, curr_text_input_mask, curr_text_sep_mask, audio_inputs, audio_mask, \
                    curr_utt_embed,vision_mask, currUtt_in_dia_idx
        elif self.args.uttORdia == 'dia':
            if self.split_type in ['train','val']:
                return curr_text_input_ids, curr_text_input_mask, curr_text_sep_mask, audio_inputs, audio_mask, \
                    curr_utt_embed,vision_mask, dia_mask, curr_label_ids,curr_numUtt_in_dia
            else:
                return curr_text_input_ids, curr_text_input_mask, curr_text_sep_mask, audio_inputs, audio_mask, \
                    curr_utt_embed,vision_mask, dia_mask, curr_numUtt_in_dia

def loading_multimodal_data(args, split=None):

    pre_output_dir = args.m3ed_project_path  
    
    m3ed = Data_Text(pre_output_dir, split, args)
    m3ed_text_features,num_utt = m3ed.preprocess_data() 

    final_data = loading_multimodal_dataset(m3ed_text_features, args, split) 

    return final_data




