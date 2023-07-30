import json
import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
import torch
import torchvision.transforms as transforms

from src.meld_bert_extraText import MELD

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')



NORMAL_MEAN = [0.5, 0.5, 0.5]
NORMAL_STD = [0.5, 0.5, 0.5]
SWIN_IMG_SIZE = 224

# MAX_DIA_LEN = 33  #(test集合中dia17的个数)

TEXT_MAX_UTT_LEN = 38  #MELD数据集中最长的utterance文本长度规定为38 其实最长有90 


import cv2
from PIL import Image
import glob


#----------------------------------------------------------------------------------------------------#

'''训练集和验证集均不做数据增强，影响挑权重''' 
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

'''加载aff-wild2数据集'''
class AffwildDataset(Dataset):
    def __init__(self, args, file_folder, anno_folder, data_list=None, transform_ops=None):
        super().__init__()

        # class_names_original = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']  #ABAW3的标注

        # label_index = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}  #MELD的标注
        class_mapping = [0, 6, 5, 2, 4, 3, 1, 7]  

        self.transforms = transform_ops 
        self.file_folder = file_folder 

        # if is_train:
        print('load Aff-Wild2_train...')
        # else:
        #     print('load Aff-Wild2_val...')

        if data_list is not None and os.path.isfile(data_list):
            print(f'  - Loading data list form: {data_list}')
            self.data_list = []
            with open(data_list, 'r') as infile:
                for line in infile:
                    self.data_list.append(
                        (line.split(' ')[0], int(line.split(' ')[1]))
                    )
        else:
            print(f'  - Generating data list form: {anno_folder}')
            save_path = args.data_list_train
            self.data_list = self.gen_list(file_folder,
                                            anno_folder,
                                            save_path=save_path,
                                            class_mapping = class_mapping)
        print(f'  - Total images: {len(self.data_list)}')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        '''load each image'''
        im = cv2.imread(os.path.join(self.file_folder, self.data_list[index][0]))
        data = Image.fromarray(im,mode='RGB')  
        data = self.transforms(data)
        label = self.data_list[index][1]

        return data, label

    def gen_list(self, file_folder, anno_folder,  save_path=None, class_mapping= None):
        """Generate list of data samples where each line contains image path and its label
            Input:
                file_folder: folder path of images (aligned)
                anno_folder: folder path of annotations, e.g., ./EXPR_Classification_Challenge/  /root/data/aff-wild2/Third ABAW Annotations
                class_mapping: list, class mapping for negative and coarse
                save_path: path of a txt file for saving list, default None
            Output:
                out_list: list of tuple contains relative file path and its label 
        """
        out_list = []
        for label_file in glob.glob(os.path.join(anno_folder, '*.txt')):   
            with open(label_file, 'r') as infile:
                print(f'----- Reading labels from: {os.path.basename(label_file)}')
                vid_name = os.path.basename(label_file)[0:-4]
                for idx, line in enumerate(infile):
                    if idx == 0:
                        classnames = line.split(',')
                    else:
                        label = int(line)  #3
                        if label == -1 or label == 7: # Remove faces with the emotions '-1' and 'other'.
                            continue
                        if class_mapping != None:
                            label = class_mapping[label]  # 
                        
                        image_name = f'{str(idx).zfill(5)}.jpg'
                        if os.path.isfile(os.path.join(file_folder, vid_name, image_name)):
                            out_list.append((os.path.join(vid_name, image_name), label)) # tuple
        if save_path is not None:
            with open(save_path, 'w') as ofile:
                for path, label in out_list:
                    ofile.write(f'{path} {label}\n')
            print(f'List saved to: {save_path}')

        return out_list


#-------------------------------------------------------------------------------------------------------------------------------------------------------


'''加载视觉单模态'''
class loading_Vision_data(Dataset):
    def __init__(self, modality_type, dataset_path, split_type):
        super(loading_Vision_data, self).__init__()
        self.modality_type = modality_type
        dataset_path = os.path.join(dataset_path, 'V', 'meld_'+split_type+'_'+modality_type+'_utt.pkl')
        dataset = pickle.load(open(dataset_path, 'rb'))
        self.modality_feature = torch.tensor(dataset[split_type][self.modality_type].astype(np.float32)).cpu().detach()  #(utt_num, utt_max_lens, extraFeature_dim)
        self.labels = torch.tensor(np.array(dataset[split_type]['labels'])).cpu().detach()   #(utt_num)
        self.utterance_mask = torch.tensor(dataset[split_type][self.modality_type+'_utt_mask']).cpu().detach() #(utt_num, utt_max_lens) 有效的置1, 无效的置0

    def get_vision_max_utt_len(self):
        return self.modality_feature.shape[1]
    
    def get_vision_featExtr_dim(self):  
        return self.modality_feature.shape[-1]    

    def get_audio_max_utt_len(self):
        return self.modality_feature.shape[1]
    
    def get_audio_featExtr_dim(self):  
        return self.modality_feature.shape[-1]   
    
    def __len__(self):
        return self.modality_feature.shape[0] 

    def __getitem__(self, index):    
        X = self.modality_feature[index]  
        utt_mask = self.utterance_mask[index]  
        Y = self.labels[index] #(1)
        return X, utt_mask, Y.cuda() 


'''加载多模态数据'''
class loading_multimodal_dataset(Dataset):
    def __init__(self, text_inputs, args, split_type):
        super(loading_multimodal_dataset, self).__init__()

        dataset_path = args.data_load_path

        self.choice_modality = args.choice_modality
        self.split_type = split_type
        '''文本模态(整个dialogue下所有utterance串起来)输入'''
        self.text_input_ids = torch.tensor([f.input_ids for f in text_inputs], dtype=torch.long).cpu().detach() #(dia_num, max_sequence_len)
        self.text_input_mask = torch.tensor([f.input_mask for f in text_inputs], dtype=torch.long).cpu().detach() #(dia_num, max_sequence_len)
        self.text_sep_mask = torch.tensor([f.sep_mask for f in text_inputs], dtype=torch.long).cpu().detach() #(dia_num, max_sequence_len) 

        '''语音模态输入'''
        audio_path = os.path.join(dataset_path, self.choice_modality, 'meld_'+split_type + '_audio_utt.pkl')
        openfile_a = open(audio_path, 'rb')
        audio_data = pickle.load(openfile_a)

        self.audio_feature = audio_data[split_type]['audio']  #(utt_num, max_audio_utt_len, extraFeature_dim) 
        self.audio_utterance_mask = audio_data[split_type]['audio_utt_mask'] #(utt_num, max_audio_utt_len) 有效的置1, 无效的置0 
        self.audio_max_utt_len, self.audio_feat_dim = self.audio_feature.shape[1], self.audio_feature.shape[-1]
        openfile_a.close()

        utt_profile_path = os.path.join(dataset_path, self.choice_modality, split_type + '_utt_profile.json')
        with open(utt_profile_path,'r') as rd:
            self.utt_profile = json.load(rd)

        '''视觉模态输入'''
        print('  - 加载人脸序列...')
        vision_path = os.path.join(dataset_path, self.choice_modality, 'meld_' + split_type + '_vision_utt.pkl')
        openfile_v = open(vision_path, 'rb')
        vision_data = pickle.load(openfile_v)
        self.vision_feature = vision_data[split_type]['vision']  #(utt_num, max_vision_utt_len, extraFeature_dim) 
        self.vision_utterance_mask = vision_data[split_type]['vision_utt_mask'] #(utt_num, max_vision_utt_len) 有效的置1, 无效的置0 
        '''三模态统一标签信息'''
        self.labels = torch.tensor(vision_data[split_type]['labels'],dtype=torch.long).cpu().detach()   #(num_utt, )  
        openfile_v.close()

        self.vision_max_utt_len, self.vision_feat_dim = self.vision_feature.shape[1], self.vision_feature.shape[-1]
        set_face_160_path = os.path.join(dataset_path, self.choice_modality, split_type+'_facseqs_160_paths_final.json')
        with open(set_face_160_path, 'r') as rr:
            self.utt_face_path = json.load(rr)
    
    def __len__(self):
        return self.vision_feature.shape[0]  

    def get_text_max_utt_len(self):
        return TEXT_MAX_UTT_LEN 

    def get_audio_max_utt_len(self):
        return self.audio_max_utt_len #平均值+3*标准差

    def get_vision_max_utt_len(self):
        return self.vision_max_utt_len  #平均值+3*标准差

    def get_audio_featExtr_dim(self):  #通过wav2vec2.0获得的768维
        return self.audio_feat_dim
    
    def get_vision_featExtr_dim(self):  #通过InceptionResnetv1获得的512维
        return self.vision_feat_dim

    def __getitem__(self, index):    

        '''
        首先得知道这个index对应是哪个utterance, 之后再去找到该dialogue的index,
        建立一个字典, key为utt_index, value为对应的utterance的名称、从属的dialogue名称, 该dialogue的编号、该utterance在该dialogue下的位置. 比如: {1:['dia0_utt0', 'dia0', 0, 3]}
        '''
        curr_utt_profile = self.utt_profile[str(index)]  
        curr_utt_name, currUtt_belg_dia_name, currUtt_belg_dia_idx, currUtt_belg_dia_len, currUtt_in_dia_idx = curr_utt_profile
        
        '''text'''
        curr_text_input_ids  = self.text_input_ids[currUtt_belg_dia_idx]
        curr_text_input_mask = self.text_input_mask[currUtt_belg_dia_idx]
        curr_text_sep_mask = self.text_sep_mask[currUtt_belg_dia_idx]

        '''audio'''
        audio_inputs = self.audio_feature[index] #加载当前utterance     .shape为(utt_max_lens, Feature_extra_dim) 
        audio_mask = self.audio_utterance_mask[index]  #(utt_max_lens)

        '''vision'''
        vision_inputs = self.vision_feature[index] #加载当前utterance     .shape为(utt_max_lens, Feature_extra_dim) 
        vision_mask = self.vision_utterance_mask[index]  #(utt_max_lens)
        vision_utt_frame_tmp = np.zeros((self.vision_max_utt_len, 3, SWIN_IMG_SIZE, SWIN_IMG_SIZE)).astype(np.float32)
        curr_utt_frm_list = self.utt_face_path[curr_utt_name]
        # print('当前utt的人脸图片个数', len(curr_utt_frm_list))
        if len(curr_utt_frm_list) > self.vision_max_utt_len:
            curr_utt_frm_list = curr_utt_frm_list[:self.vision_max_utt_len] #对大于MAX_UTT_LEN的图片进行截断
        # curr_utt_len = len(curr_utt_frm_list) 

        '''获得每张帧级别人脸准备过Swin模型'''
        curr_vision_utt_feat_src_task = from_image_to_embedding_no_IncepRes(curr_utt_frm_list,self.split_type) #(curr_utt_frame_len, 3, 224, 224)
        cur_vis_num_imgs = curr_vision_utt_feat_src_task.shape[0]
        for jj in range(curr_vision_utt_feat_src_task.shape[0]):
            vision_utt_frame_tmp[jj,:] = curr_vision_utt_feat_src_task[jj,:]
        curr_vision_utt_feat_src_task = vision_utt_frame_tmp #(utt_frame_max_len, 3, 224, 224)

        '''label'''
        curr_label_ids = self.labels[index] 
        return curr_text_input_ids, curr_text_input_mask, curr_text_sep_mask, audio_inputs, audio_mask, \
                        vision_inputs, vision_mask, curr_label_ids, curr_vision_utt_feat_src_task, cur_vis_num_imgs, currUtt_in_dia_idx


def loading_multimodal_data(args, split=None):

    load_anno_csv = args.load_anno_csv_path

    meld_text_path = args.meld_text_path

    meld = MELD(load_anno_csv, args.pretrainedtextmodel_path, meld_text_path, split)

    meld_text_features = meld.preprocess_data()  

    final_data = loading_multimodal_dataset(meld_text_features, args, split) 

    return final_data




