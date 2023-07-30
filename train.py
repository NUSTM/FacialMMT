import torch
from torch import nn
from src.models import SwinForAffwildClassification, MultiModalTransformerForClassification, meld_utt_transformer
from utils.util import *
import transformers
import time
from utils.eval_metrics import eval_meld
from pytorch_lightning.lite import LightningLite
from time import strftime

class Lite(LightningLite):
    def run(self, args, aux_train_loader, trg_train_loader, trg_valid_loader, trg_test_loader):
        
        #----------------------------------------------------define training on auxiliary task---------------------------------------------------------#
        def aux_train(self, shareSwin_model, share_model_optimizer, share_model_scheduler, criterion):
            shareSwin_model.train()
            num_batches = args.aux_n_train // args.aux_batch_size      #num_of_imgs // batch_size == len(aux_train_loader) 
            total_loss, total_size = 0, 0
            start_time = time.time()
            share_model_optimizer.zero_grad()
            for i_batch, (image_feature, labels) in enumerate(aux_train_loader):    #
                image_feature,  labels  =  image_feature.cuda(),  labels.cuda()
                labels = labels.long()
                batch_size = args.aux_batch_size 
                loss = 0
                loss = shareSwin_model(image_feature, False, labels, criterion)  
                loss = loss/ args.aux_accumulation_steps
                self.backward(loss)
                if ((i_batch+1) % args.aux_accumulation_steps)==0:
                    torch.nn.utils.clip_grad_norm_(shareSwin_model.parameters(), args.clip)
                    share_model_optimizer.step() 
                    share_model_scheduler.step()
                    share_model_optimizer.zero_grad()
                total_loss += loss.item() * batch_size * args.aux_accumulation_steps
                total_size += batch_size #
                if i_batch % args.aux_log_interval == 0 and i_batch > 0:   
                    avg_loss = total_loss / total_size
                    elapsed_time = time.time() - start_time
                    print('**SRC** | Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                        format(epoch, i_batch, num_batches, elapsed_time * 1000 / args.aux_log_interval, avg_loss)) 
                    total_loss, total_size = 0, 0
                    start_time = time.time()
        
        #-----------------------------------------------------define training and evaluation on main task-------------------------------------------------------#
        
        def multimodal_train(self, shareSwin_model, multimodal_model, multimodal_model_optimizer, multimodal_model_scheduler, criterion):
            shareSwin_model.train()
            multimodal_model.train()
            num_batches = args.trg_n_train // args.trg_batch_size      #num_of_utt // batch_size == len(trg_train_loader) 
            total_loss, total_size = 0, 0
            start_time = time.time()
            multimodal_model_optimizer.zero_grad()

            for i_batch, batch in enumerate(trg_train_loader):   
                batch_size = args.trg_batch_size 
                loss = 0
                batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, vision_inputs, \
                        vision_mask, batch_label_ids, batch_vision_utt_feat_aux_task, batch_vis_num_imgs, batchUtt_in_dia_idx = batch

                '''determine the total number of faces for the given utterances'''
                batch_img_feat_list = []
                utt_idx = 0
                for utt_idx in range(len(vision_inputs)):
                    curr_utt_num_imgs = batch_vis_num_imgs[utt_idx]
                    curr_utt_img_feat = batch_vision_utt_feat_aux_task[utt_idx][:curr_utt_num_imgs]
                    batch_img_feat_list.append(curr_utt_img_feat)
                    utt_idx += 1
                batch_img_feat_temp = batch_img_feat_list[0] #count the number of faces in the first utterance.
                if len(batch_img_feat_list) > 1:
                    for i in range(1,len(batch_img_feat_list)):
                        batch_img_feat_temp = torch.cat((batch_img_feat_temp,batch_img_feat_list[i]),dim=0)  #(batch_num_img, 3, 224, 224)

                preds_FacialEmo_distr = shareSwin_model(batch_img_feat_temp,is_trg_task=True) #(batch_num_img, 7) 

                '''Calculate the confidence score for each face expression and filter out those with a confidence score less than 0.2. 
                        If the expressions are evenly distributed, the result should be 0.14 * 0.14 * 7 = 0.137.'''
                batch_FacialEmo_import_matrix = torch.mm(preds_FacialEmo_distr,preds_FacialEmo_distr.t())    
                batch_FacialEmo_import = torch.diagonal(batch_FacialEmo_import_matrix) 
                batch_FacialEmo_import_mask = batch_FacialEmo_import.gt(args.FacialEmoImpor_threshold)  
                batch_HighImporFace_idx = torch.nonzero(batch_FacialEmo_import_mask).squeeze(1)  
                batch_vis_emo = torch.zeros([vision_inputs.shape[0], vision_inputs.shape[1], args.num_labels])

                '''Note: can't filter out all the faces.'''
                if len(batch_HighImporFace_idx) > 0:
                    temp_batch_HighImporFace_idx = batch_HighImporFace_idx
                    new_vision_mask = torch.zeros((vision_mask.shape))
                    margin = 0
                    for utt_idx in range(len(vision_inputs)):
                        real_batch_idx = 0
                        for img_idx in range(len(temp_batch_HighImporFace_idx)):
                            if temp_batch_HighImporFace_idx[img_idx] < batch_vis_num_imgs[utt_idx] + margin:  
                                new_vision_mask[utt_idx][real_batch_idx] = 1
                                real_batch_idx += 1
                            else:
                                break
                        #Remove the faces we've assigned
                        margin = margin + batch_vis_num_imgs[utt_idx] -1
                        temp_batch_HighImporFace_idx = temp_batch_HighImporFace_idx[real_batch_idx:]

                    new_vision_inputs = torch.zeros((vision_inputs.shape))
                    jj=0
                    margin = 0
                    for utt_idx in range(len(vision_inputs)):
                        for fram_idx in range(len(new_vision_mask[utt_idx])):
                            if new_vision_mask[utt_idx][fram_idx] !=0:
                                batch_vis_emo[utt_idx][fram_idx] = preds_FacialEmo_distr[batch_HighImporFace_idx[jj]]
                                new_vision_inputs[utt_idx][fram_idx] = vision_inputs[utt_idx][batch_HighImporFace_idx[jj]-margin]
                                jj+=1
                            else:
                                break
                        margin = margin + batch_vis_num_imgs[utt_idx]-1  

                    '''表情嵌入拼接到更新后的视觉嵌入的后面'''
                    vision_inputs_concat = torch.cat((new_vision_inputs, batch_vis_emo), dim=-1)
                    #for memory
                    del vision_inputs,vision_mask,batch_vis_emo,new_vision_inputs,temp_batch_HighImporFace_idx,batch_FacialEmo_import,batch_FacialEmo_import_mask,\
                        batch_HighImporFace_idx, batch_img_feat_list, batch_img_feat_temp, batch_vision_utt_feat_aux_task,preds_FacialEmo_distr
                    
                    logits = multimodal_model(batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, \
                                                    vision_inputs_concat, new_vision_mask, batchUtt_in_dia_idx) 

                elif len(batch_HighImporFace_idx) == 0:
                    jj = 0
                    for i in range(len(vision_inputs)): 
                        for j in range(len(vision_inputs[i])):
                            if vision_mask[i][j] == 1:
                                batch_vis_emo[i][j] = preds_FacialEmo_distr[jj]
                                jj += 1
                            else:
                                break
                    vision_inputs_concat = torch.cat((vision_inputs, batch_vis_emo), dim=-1)
                    logits = multimodal_model(batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, \
                                                    vision_inputs_concat, vision_mask, batchUtt_in_dia_idx) 

                loss = criterion(logits, batch_label_ids.cuda())
                loss = loss/ args.trg_accumulation_steps
                self.backward(loss)
                
                if ((i_batch+1)%args.trg_accumulation_steps)==0:
                    torch.nn.utils.clip_grad_norm_(multimodal_model.parameters(), args.clip)
                    multimodal_model_optimizer.step() 
                    multimodal_model_scheduler.step()
                    multimodal_model_optimizer.zero_grad()
                total_loss += loss.item() * batch_size * args.trg_accumulation_steps
                total_size += batch_size 
                if i_batch % args.trg_log_interval == 0 and i_batch > 0:   
                    avg_loss = total_loss / total_size
                    elapsed_time = time.time() - start_time
                    print('**TRG** | Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                        format(epoch, i_batch, num_batches, elapsed_time * 1000 / args.trg_log_interval, avg_loss))  
                    total_loss, total_size = 0, 0
                    start_time = time.time()

        def multimodal_evaluate(shareSwin_model, multimodal_model, criterion, test=False):

            shareSwin_model.eval()
            multimodal_model.eval()
            loader = trg_test_loader if test else trg_valid_loader
            total_loss = 0.0
            results = []
            truths = []

            with torch.no_grad():
                for i_batch, batch in enumerate(loader):    
                    batch_size = args.trg_batch_size 
                    batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, vision_inputs, \
                            vision_mask, batch_label_ids, batch_vision_utt_feat_aux_task, batch_vis_num_imgs, batchUtt_in_dia_idx = batch

                    batch_img_feat_list = []
                    utt_idx = 0
                    for utt_idx in range(len(vision_inputs)):
                        curr_utt_num_imgs = batch_vis_num_imgs[utt_idx]
                        curr_utt_img_feat = batch_vision_utt_feat_aux_task[utt_idx][:curr_utt_num_imgs]
                        batch_img_feat_list.append(curr_utt_img_feat)
                        utt_idx += 1
                    batch_img_feat_temp = batch_img_feat_list[0] 
                    if len(batch_img_feat_list) > 1:
                        for i in range(1,len(batch_img_feat_list)):
                            batch_img_feat_temp = torch.cat((batch_img_feat_temp,batch_img_feat_list[i]),dim=0)  

                    preds_FacialEmo_distr = shareSwin_model(batch_img_feat_temp,is_trg_task=True) 

                    batch_FacialEmo_import_matrix = torch.mm(preds_FacialEmo_distr,preds_FacialEmo_distr.t())    
                    batch_FacialEmo_import = torch.diagonal(batch_FacialEmo_import_matrix) 
                    batch_FacialEmo_import_mask = batch_FacialEmo_import.gt(args.FacialEmoImpor_threshold)  
                    batch_HighImporFace_idx = torch.nonzero(batch_FacialEmo_import_mask).squeeze(1)  
                    batch_vis_emo = torch.zeros([vision_inputs.shape[0], vision_inputs.shape[1], args.num_labels])
                    if len(batch_HighImporFace_idx) > 0:
                        temp_batch_HighImporFace_idx = batch_HighImporFace_idx
                        new_vision_mask = torch.zeros((vision_mask.shape))
                        margin = 0
                        for utt_idx in range(len(vision_inputs)): # 0 1 2
                            real_batch_idx = 0
                            for img_idx in range(len(temp_batch_HighImporFace_idx)):
                                if temp_batch_HighImporFace_idx[img_idx] < batch_vis_num_imgs[utt_idx] + margin:  #29 
                                    new_vision_mask[utt_idx][real_batch_idx] = 1
                                    real_batch_idx += 1
                                else:
                                    break
                            margin = margin + batch_vis_num_imgs[utt_idx] -1
                            temp_batch_HighImporFace_idx = temp_batch_HighImporFace_idx[real_batch_idx:]
                        new_vision_inputs = torch.zeros((vision_inputs.shape))
                        jj=0
                        margin = 0
                        for utt_idx in range(len(vision_inputs)):
                            for fram_idx in range(len(new_vision_mask[utt_idx])):
                                if new_vision_mask[utt_idx][fram_idx] !=0:
                                    batch_vis_emo[utt_idx][fram_idx] = preds_FacialEmo_distr[batch_HighImporFace_idx[jj]]
                                    new_vision_inputs[utt_idx][fram_idx] = vision_inputs[utt_idx][batch_HighImporFace_idx[jj]-margin]
                                    jj+=1
                                else:
                                    break
                            margin = margin + batch_vis_num_imgs[utt_idx]-1 #6

                        vision_inputs_concat = torch.cat((new_vision_inputs, batch_vis_emo), dim=-1)
                        #for memory
                        del vision_inputs,vision_mask,batch_vis_emo,new_vision_inputs,temp_batch_HighImporFace_idx,batch_FacialEmo_import,batch_FacialEmo_import_mask,\
                            batch_HighImporFace_idx, batch_img_feat_list, batch_img_feat_temp, batch_vision_utt_feat_aux_task,preds_FacialEmo_distr
                        
                        logits = multimodal_model(batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, \
                                                        vision_inputs_concat, new_vision_mask, batchUtt_in_dia_idx) 

                    elif len(batch_HighImporFace_idx) == 0:
                        jj = 0
                        for i in range(len(vision_inputs)): 
                            for j in range(len(vision_inputs[i])):
                                if vision_mask[i][j] == 1:
                                    batch_vis_emo[i][j] = preds_FacialEmo_distr[jj]
                                    jj += 1
                                else:
                                    break
                        vision_inputs_concat = torch.cat((vision_inputs, batch_vis_emo), dim=-1)
                        logits = multimodal_model(batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, \
                                                        vision_inputs_concat, vision_mask, batchUtt_in_dia_idx) 

                    total_loss += criterion(logits, batch_label_ids.cuda()).item() * batch_size
                    # Collect the results into dictionary
                    results.append(logits)  #
                    truths.append(batch_label_ids) 
            avg_loss = total_loss / (args.trg_n_test if test else args.trg_n_valid)
            results = torch.cat(results)
            truths = torch.cat(truths)
            return avg_loss, results, truths

        def unimodal_train(self, unimodal_model, optimizer, scheduler, criterion):
            unimodal_model.train()
            num_batches = args.trg_n_train // args.trg_batch_size     
            total_loss, total_size = 0, 0
            start_time = time.time()
            optimizer.zero_grad()
            for i_batch, batch in enumerate(trg_train_loader):    
                batch_size = args.trg_batch_size 
                loss = 0
                modality_feature, utterance_mask, labels = batch
                logits = unimodal_model(modality_feature, utterance_mask)  
                loss = criterion(logits, labels)
                loss = loss/ args.trg_accumulation_steps
                self.backward(loss)
                if ((i_batch+1)%args.trg_accumulation_steps)==0:
                    torch.nn.utils.clip_grad_norm_(unimodal_model.parameters(), args.clip)
                    optimizer.step() 
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * batch_size * args.trg_accumulation_steps
                total_size += batch_size 
                if i_batch % args.trg_log_interval == 0 and i_batch > 0:   
                    avg_loss = total_loss / total_size
                    elapsed_time = time.time() - start_time
                    print('**TRG** | Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                        format(epoch, i_batch, num_batches, elapsed_time * 1000 / args.trg_log_interval, avg_loss))  
                    total_loss, total_size = 0, 0
                    start_time = time.time()

        def unimodal_evaluate(unimodal_model, criterion, test=False):
            unimodal_model.eval()
            loader = trg_test_loader if test else trg_valid_loader
            total_loss = 0.0
            results = []
            truths = []
            with torch.no_grad():
                for i_batch, batch in enumerate(loader):    
                    batch_size = args.trg_batch_size 
                    modality_feature, utterance_mask, labels = batch
                    logits = unimodal_model(modality_feature, utterance_mask)  
                    total_loss += criterion(logits, labels).item() * batch_size
                    results.append(logits)  
                    truths.append(labels) 
            avg_loss = total_loss / (args.trg_n_test if test else args.trg_n_valid)
            results = torch.cat(results)
            truths = torch.cat(truths)
            return avg_loss, results, truths
        
        #-------------------------------------------------------------------------------------------------------------------------------#
        criterion = nn.CrossEntropyLoss()
        #training from scratch
        if not args.doEval:
            if args.choice_modality == 'T+A+V':
                aux_train_loader = self.setup_dataloaders(aux_train_loader, move_to_device=False)  # Scale your dataloaders
            trg_train_loader = self.setup_dataloaders(trg_train_loader, move_to_device=False)
            trg_valid_loader = self.setup_dataloaders(trg_valid_loader, move_to_device=False)
            trg_test_loader  = self.setup_dataloaders(trg_test_loader, move_to_device=False)

            #------------------------------------------------------------Loading various unimodal or multimodal models and optimizers-----------------------------------------------------------#
            if args.choice_modality == 'T+A+V':
                multimodal_model = MultiModalTransformerForClassification(args)
                multimodal_model_optimizer = transformers.AdamW(multimodal_model.parameters(), lr=args.trg_lr,weight_decay=args.weight_decay,no_deprecation_warning=True)  
                multimodal_model, multimodal_model_optimizer = self.setup(multimodal_model, multimodal_model_optimizer)  # Scale your model / optimizers
                multimodal_total_training_steps = args.num_epochs * len(trg_train_loader) / args.trg_accumulation_steps 
                multimodal_model_scheduler = transformers.get_linear_schedule_with_warmup(optimizer = multimodal_model_optimizer,
                                                                num_warmup_steps = int(multimodal_total_training_steps * args.warm_up),
                                                                num_training_steps = multimodal_total_training_steps)

                shareSwin_model = SwinForAffwildClassification(args)
                '''loading the pretrained SWIN (Swin Transformer) model and updating the parameter dictionary.'''
                model_dict = shareSwin_model.state_dict()
                pretrained_dict = torch.load(args.pretrained_backbone_path)['state_dict']
                new_pretrained_dict = {}
                for k in model_dict:
                    if k in pretrained_dict:
                        if k == 'classifier.weight':
                            continue
                        if k == 'classifier.bias':
                            continue
                        if k[:5] == 'swin.':
                            k_val = k[5:]
                        else:
                            k_val = k
                        new_pretrained_dict[k] = pretrained_dict['backbone.' + k_val] # tradition training
                model_dict.update(new_pretrained_dict)
                shareSwin_model.load_state_dict(model_dict)
                
                share_model_optimizer = transformers.AdamW(shareSwin_model.parameters(), lr=args.aux_lr)  

                shareSwin_model, share_model_optimizer = self.setup(shareSwin_model, share_model_optimizer)  # Scale your model / optimizers

                share_total_training_steps = args.num_epochs * len(aux_train_loader) / args.aux_accumulation_steps 
                share_model_scheduler = transformers.get_linear_schedule_with_warmup(optimizer = share_model_optimizer,
                                                                    num_warmup_steps = int(share_total_training_steps * args.warm_up),
                                                                    num_training_steps = share_total_training_steps)

            elif args.choice_modality == 'V':
                unimodal_model = meld_utt_transformer(args)
                optimizer = transformers.AdamW(unimodal_model.parameters(), lr=args.trg_lr, weight_decay=args.weight_decay, no_deprecation_warning=True)  
                unimodal_model, optimizer = self.setup(unimodal_model, optimizer)  # Scale your model / optimizers
                total_training_steps = args.num_epochs * len(trg_train_loader) / args.trg_accumulation_steps 
                scheduler = transformers.get_linear_schedule_with_warmup(optimizer = optimizer,
                                                                        num_warmup_steps = int(total_training_steps * args.warm_up),
                                                                        num_training_steps = total_training_steps)
            
            #---------------------------------------------------------------------Adjust and optimize model-----------------------------------------------------------------------#
            best_valid_f1 = 0
            best_model_time = 0
            for epoch in range(1, args.num_epochs+1): 
                if args.choice_modality == 'T+A+V':
                    #----------------------------------------------------auxiliary task---------------------------------------------------------#
                    start = time.time()
                    aux_train(self, shareSwin_model, share_model_optimizer, share_model_scheduler, criterion)  
                    end = time.time()
                    duration = end-start
                    print("-"*50)
                    print('**SRC** | Epoch {:2d} | Time {:5.4f} hour'.format(epoch, duration/3600))
                    print("-"*50)
                    #-----------------------------------------------------target task--------------------------------------------------------#
                    start = time.time()
                    multimodal_train(self, shareSwin_model, multimodal_model, multimodal_model_optimizer, multimodal_model_scheduler, criterion)
                    val_loss, results, truths = multimodal_evaluate(shareSwin_model, multimodal_model, criterion, test=False)
                    end = time.time()
                    duration = end-start

                    val_wg_av_f1 = eval_meld(results, truths, test=False)
                    print("-"*50)
                    print('**TRG** | Epoch {:2d} | Time {:5.4f} hour | val_wg_av_f1 {:5.4f} '.format(epoch, duration/3600, val_wg_av_f1))
                    print("-"*50)

                    #save the best model on validation set
                    if val_wg_av_f1 > best_valid_f1:
                        current_time = strftime("%m-%d-%H-%M-%S")
                        second_multimodel_path = os.path.join(args.save_Model_path,'multimodal_model_{}_{}.pt'.format(args.choice_modality, best_model_time))
                        if os.path.exists(second_multimodel_path):
                            os.remove(second_multimodel_path)
                        second_shareSwin_path = os.path.join(args.save_Model_path,'best_swin_{}.pt'.format(best_model_time))
                        if os.path.exists(second_shareSwin_path):
                            os.remove(second_shareSwin_path)
                        save_Swin_model(shareSwin_model,args, current_time)
                        save_Multimodal_model(multimodal_model,args, current_time)
                        best_valid_f1 = val_wg_av_f1
                        best_model_time = current_time

                elif args.choice_modality == 'V':
                    start = time.time()
                    unimodal_train(self, unimodal_model, optimizer, scheduler, criterion)
                    _, results, truths = unimodal_evaluate(unimodal_model, criterion, test=False)
                    end = time.time()
                    duration = end-start
                    val_wg_av_f1 = eval_meld(results, truths, test=False)
                    print("-"*50)
                    print('**TRG** | Epoch {:2d} | Time {:5.4f} min | val_wg_av_f1 {:5.4f} '.format(epoch, duration/60, val_wg_av_f1))
                    print("-"*50)

                    #save the best model on validation set
                    if val_wg_av_f1 > best_valid_f1:
                        current_time = strftime("%m-%d-%H-%M-%S")
                        second_model_path = os.path.join(args.save_Model_path,'unimodal_model_{}_{}.pt'.format(args.choice_modality, best_model_time))
                        if os.path.exists(second_model_path):
                            os.remove(second_model_path)
                        save_Unimodal_model(unimodal_model,args, current_time)
                        best_valid_f1 = val_wg_av_f1
                        best_model_time = current_time
        #---------------------------------------------------------------------test----------------------------------------------------------------------------#
            #conduct evaluation on the best model
            print("&"*50)
            if args.choice_modality == 'T+A+V' :
                best_share_model = load_Swin_model(args, best_model_time)
                best_multi_model = load_Multimodal_model(args.choice_modality, args.save_Model_path, best_model_time)
                _, results, truths = multimodal_evaluate(best_share_model, best_multi_model, criterion, test=True)
            elif args.choice_modality == 'V':
                best_unimodal_model = load_Unimodal_model(args.choice_modality, args.save_Model_path, best_model_time)
                _, results, truths = unimodal_evaluate(best_unimodal_model, criterion, test=True)
            print('**TEST** | wg_av_f1 {:5.4f} '.format(eval_meld(results, truths, test=True)))
            print('\n')

        #conduct evaluation directly without training
        elif args.doEval:
            load_project_path = os.path.abspath(os.path.dirname(__file__))
            print("&"*50)
            if args.choice_modality == 'T+A+V' :
                best_share_model = torch.load(os.path.join(load_project_path, 'pretrained_model', args.load_swin_path))
                best_multi_model = torch.load(os.path.join(load_project_path, 'pretrained_model', args.load_multimodal_path))
                _, results, truths = multimodal_evaluate(best_share_model, best_multi_model, criterion, test=True)
            elif args.choice_modality == 'V':
                best_unimodal_model = torch.load(os.path.join(load_project_path, 'pretrained_model', args.load_unimodal_path))
                _, results, truths = unimodal_evaluate(best_unimodal_model, criterion, test=True)
            print('**TEST** | wg_av_f1 {:5.4f} '.format(eval_meld(results, truths, test=True)))
            print('\n')
