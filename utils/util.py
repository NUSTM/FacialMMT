import torch
import os
from utils.dataset import loading_multimodal_data, loading_Vision_data

import random
import numpy as np
import PIL
import torchvision.transforms as transforms

from utils.random_erasing import RandomErasing
from torchvision.transforms import InterpolationMode
from utils.dataset import AffwildDataset
import yaml


NORMAL_MEAN = [0.5, 0.5, 0.5]
NORMAL_STD = [0.5, 0.5, 0.5]
SWIN_IMG_SIZE = 224

#---------------------------------------------------------------------------------------------------------------------------------------------#
#aff-wild2数据集加载
class RandomApply():
    def __init__(self, transforms, prob=0.5):
        self.prob = prob
        self.transforms = transforms
    def __call__(self, x):
        if random.random() > self.prob:
            for t in self.transforms:
                x = t(x)
        return x

class GaussianBlur():
    def __init__(self, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, x):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        x = x.filter(PIL.ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_train_transforms(backbone_param):
    aug_op_list = []
    scale_size = int(backbone_param['img_size'])
    aug_op_list.append(transforms.Resize(scale_size, interpolation=InterpolationMode.BICUBIC))
    aug_op_list.append(RandomApply([transforms.Grayscale(3)], prob=0.2))
    aug_op_list.append(RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], prob=0.8))
    aug_op_list.append(RandomApply([GaussianBlur(0.1, 2.0)], prob=0.5))
    aug_op_list.append(transforms.ToTensor())
    aug_op_list.append(transforms.Normalize(mean=NORMAL_MEAN,
                                            std=NORMAL_STD))

    random_erasing = RandomErasing(prob=0.25,
                                    mode='pixel',
                                    max_count=1,
                                    num_splits=False)
    aug_op_list.append(random_erasing)
    transforms_train = transforms.Compose(aug_op_list)
    return transforms_train


# def get_val_transforms(backbone_param):
#     scale_size = int(backbone_param['img_size'])
#     transforms_val = transforms.Compose([
#         transforms.Resize(scale_size, interpolation=InterpolationMode.BICUBIC), #需要将aff-wild2的112的size放大至224
#         transforms.CenterCrop((backbone_param['img_size'], backbone_param['img_size'])),
#         transforms.ToTensor(),
#         transforms.Normalize(mean = NORMAL_MEAN, std=NORMAL_STD)])
#     return transforms_val

def get_affwild2_dataset(args):
    with open(args.backbone_conf_file) as f:
            backbone_conf = yaml.load(f, Loader=yaml.FullLoader)
            backbone_param = backbone_conf[args.backbone_type]

    # if is_train:
    transform_ops = get_train_transforms(backbone_param)
    # else:
    #     transform_ops = get_val_transforms(backbone_param)
    dataset = AffwildDataset(args, file_folder=args.data_folder,
                            anno_folder=args.anno_folder,
                            data_list=args.data_list_train,
                            # is_train=is_train,
                            transform_ops=transform_ops)
    return dataset

#---------------------------------------------------------------------------------------------------------------------------------------------#

def get_multimodal_data(args, split='train'):
    data_path = os.path.join(args.data_load_path, args.choice_modality, f'meld_multimodal_{split}_{args.choice_modality}_{args.plm_name}.dt')
    print('load MELD_multimodal_'+args.choice_modality+'_'+split+'...')
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = loading_multimodal_data(args,split)  
        torch.save(data, data_path, pickle_protocol=4)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path, map_location=torch.device('cpu'))
    return data

#---------------------------------------------------------------------------------------------------------------------------------------------#

def get_meld_vision(args, split='train'):
    data_path = os.path.join(args.data_load_path, args.choice_modality, f'meld_unimodal_{split}_{args.choice_modality}.dt')
    print('load MELD_unimodal_'+args.choice_modality+'_'+split+'...')
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        modality_type = 'vision'
        data = loading_Vision_data(modality_type, args.data_load_path, split)  #
        torch.save(data, data_path, pickle_protocol=4)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path, map_location=torch.device('cpu'))
    return data

#---------------------------------------------------------------------------------------------------------------------------------------------#

load_project_path = os.path.abspath(os.path.dirname(__file__))

def save_Swin_model(model,args,curr_time):

    save_model_name = 'best_swin_{}.pt'.format(curr_time)
    sve_path = os.path.join(args.save_Model_path, save_model_name)
    torch.save(model, sve_path, pickle_protocol=4)
    print(f"Saved model at saved_model/{save_model_name}!")

def save_Multimodal_model(model, args,curr_time):
    save_model_name = 'multimodal_model_{}_{}.pt'.format(args.choice_modality,curr_time)
    save_path = os.path.join(args.save_Model_path, save_model_name)
    torch.save(model, save_path, pickle_protocol=4)
    print(f"Saved model at saved_model/{save_model_name}!")

def load_Swin_model(args,best_model_time):
    save_model_name = 'best_swin_{}.pt'.format(best_model_time)
    load_path = os.path.join(args.save_Model_path, save_model_name)
    print('Loading the best Swin model for testing:'+save_model_name)
    model = torch.load(load_path)
    return model

def load_Multimodal_model(choice_modality, save_Model_path,best_model_time):
    save_model_name = 'multimodal_model_{}_{}.pt'.format(choice_modality,best_model_time)
    load_path = os.path.join(save_Model_path, save_model_name)
    print('Loading the best Multimodal model for testing:'+save_model_name)
    model = torch.load(load_path)
    return model

#---------------------------------------------------------------------------------------------------------------------------------------------#
def save_Unimodal_model(model, args,curr_time):
    save_model_name = os.path.join(args.save_Model_path, 'unimodal_model_{}_{}.pt').format(args.choice_modality,curr_time)
    torch.save(model, save_model_name, pickle_protocol=4)
    print(f"Saved model at saved_model/unimodal_model_{args.choice_modality}_{curr_time}.pt!")

def load_Unimodal_model(choice_modality, save_Model_path,best_model_time):
    save_model_name = 'unimodal_model_{}_{}.pt'.format(choice_modality,best_model_time)
    load_path = os.path.join(save_Model_path, save_model_name)
    print('Loading the best unimodal model for testing:'+save_model_name)
    model = torch.load(load_path)
    return model

