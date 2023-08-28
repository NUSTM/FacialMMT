import torch
from torch import nn
from src.models import MultiModalTransformerForClassification_utt_level,MultiModalTransformerForClassification_dia_level
from utils.util import *
import transformers
import time
from utils.eval_metrics import eval_m3ed
from pytorch_lightning.lite import LightningLite
from time import strftime


class Lite(LightningLite):
    def run(self, hyp_params, trg_train_loader, trg_valid_loader, trg_test_loader):
        
        #-----------------------------------------------------定义在目标领域m3ed上的训练和评估--------------------------------------------------------#
        #定义目标域多模态上的训练
        def multimodal_train(self, multimodal_model, multimodal_model_optimizer, multimodal_model_scheduler, criterion):
            multimodal_model.train()
            num_batches = hyp_params.trg_n_train // hyp_params.trg_batch_size      #num_of_utt // batch_size == len(trg_train_loader) 
            total_loss, total_size = 0, 0
            start_time = time.time()
            multimodal_model_optimizer.zero_grad()

            for i_batch, batch in enumerate(trg_train_loader):    
                batch_size = hyp_params.trg_batch_size 
                loss = 0
                if hyp_params.uttORdia == 'utt':
                    batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, \
                        vision_inputs, vision_mask, batch_label_ids, batchUtt_in_dia_idx = batch
                    logits = multimodal_model(batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, \
                                            vision_inputs, vision_mask, batchUtt_in_dia_idx)  
                elif hyp_params.uttORdia == 'dia':
                    batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, \
                        vision_inputs, vision_mask, dia_mask, batch_label_ids,curr_numUtt_in_dia = batch
                    logits = multimodal_model(batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, \
                                            vision_inputs, vision_mask, dia_mask,curr_numUtt_in_dia)

                truely_label = batch_label_ids.masked_select(dia_mask.bool()).squeeze().cuda()
                loss = criterion(logits, truely_label)

                loss = loss/ hyp_params.trg_accumulation_steps

                self.backward(loss)

                if ((i_batch+1)%hyp_params.trg_accumulation_steps)==0:
                    torch.nn.utils.clip_grad_norm_(multimodal_model.parameters(), hyp_params.clip)
                    multimodal_model_optimizer.step() 
                    multimodal_model_scheduler.step()
                    multimodal_model_optimizer.zero_grad()

                total_loss += loss.item() * batch_size * hyp_params.trg_accumulation_steps

                total_size += batch_size 
                if i_batch % hyp_params.trg_log_interval == 0 and i_batch > 0:   
                    avg_loss = total_loss / total_size
                    elapsed_time = time.time() - start_time
                    print('**TRG** | Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                        format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.trg_log_interval, avg_loss))  #每迭代trg_log_interval个样本打印一次结果
                    total_loss, total_size = 0, 0
                    start_time = time.time()


        #定义目标域多模态上的评估
        def multimodal_evaluate(multimodal_model, criterion, test=False):
            multimodal_model.eval()
            loader = trg_test_loader if test else trg_valid_loader
            total_loss = 0.0
            results = []
            truths = []
            with torch.no_grad():
                for i_batch, batch in enumerate(loader):    
                    batch_size = hyp_params.trg_batch_size
                    if hyp_params.uttORdia == 'utt':
                        batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, \
                                vision_inputs, vision_mask, batch_label_ids, batchUtt_in_dia_idx = batch
                        logits = multimodal_model(batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, \
                                                    vision_inputs, vision_mask, batchUtt_in_dia_idx)  
                    elif hyp_params.uttORdia == 'dia':
                        batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, \
                                vision_inputs, vision_mask, dia_mask, batch_label_ids,curr_numUtt_in_dia = batch
                        logits = multimodal_model(batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, \
                                                    vision_inputs, vision_mask, dia_mask,curr_numUtt_in_dia)

                    truely_label = batch_label_ids.masked_select(dia_mask.bool()).squeeze().cuda()
                    total_loss += criterion(logits, truely_label).item() * batch_size

                    # Collect the results into dictionary
                    results.append(logits)  #
                    truths.append(truely_label) 

            avg_loss = total_loss / (hyp_params.trg_n_test if test else hyp_params.trg_n_valid)

            results = torch.cat(results)
            truths = torch.cat(truths)
            return avg_loss, results, truths

        #------------------------------------------------------------------------加载目标域数据并训练---------------------------------------------------------#
        if not hyp_params.conduct_emo_eval:
            trg_train_loader = self.setup_dataloaders(trg_train_loader, move_to_device=False)
            trg_valid_loader = self.setup_dataloaders(trg_valid_loader, move_to_device=False)
            #------------------------------------------------------------加载各种单模态或者多模态模型、优化器-----------------------------------------------------------#

            multimodal_model = MultiModalTransformerForClassification_utt_level(hyp_params) if hyp_params.uttORdia == 'utt' else MultiModalTransformerForClassification_dia_level(hyp_params)
            multimodal_model_optimizer = transformers.AdamW(multimodal_model.parameters(), lr=hyp_params.trg_lr,weight_decay=hyp_params.weight_decay)  
            multimodal_model, multimodal_model_optimizer = self.setup(multimodal_model, multimodal_model_optimizer)  # Scale your model / optimizers
            criterion = nn.CrossEntropyLoss()
            multimodal_total_training_steps = hyp_params.num_epochs * len(trg_train_loader) / hyp_params.trg_accumulation_steps 
            multimodal_model_scheduler = transformers.get_linear_schedule_with_warmup(optimizer = multimodal_model_optimizer,
                                                        num_warmup_steps = int(multimodal_total_training_steps * hyp_params.warm_up),
                                                        num_training_steps = multimodal_total_training_steps)
            best_valid_f1 = 0
            best_model_time = 0

            best_val_loss = np.inf  # 初始最好的验证损失，设为无穷大
            patience = hyp_params.patience  # 连续多少个epoch验证损失不下降后停止训练
            counter = 0  # 连续上升的epoch计数器

            for epoch in range(1, hyp_params.num_epochs+1):  #循环epoch次
                
                if hyp_params.choice_modality in ('T+A', 'T+V', 'T+A+V'):
                    if hyp_params.add_or_not_emo_embed or hyp_params.choice_modality == 'T+A': 
                        start = time.time()
                        multimodal_train(self, multimodal_model, multimodal_model_optimizer, multimodal_model_scheduler, criterion)
                        val_loss, results, truths = multimodal_evaluate(multimodal_model, criterion, test=False)
                        end = time.time()
                        duration = end-start

                    macro_f1 = eval_m3ed(results, truths, hyp_params.choice_modality, test=False)
                    print("-"*50)
                    print('**TRG** | Epoch {:2d} | Time {:5.4f} min | Val_Loss {:5.4f} | macro_f1 {:5.4f} '.format(epoch, duration/60, val_loss, macro_f1))
                    print("-"*50)
                    
                    #保存中间最佳模型
                    if macro_f1 > best_valid_f1:
                        current_time = strftime("%m-%d-%H-%M-%S")
                        print(f"Saved model at save/multimodal_transformer.pt!")

                        second_model_path = os.path.join('/home/devin/CCAC/FacialMMTforM3ED/save','multimodal_transformer_{}_{}.pt'.format(hyp_params.choice_modality, best_model_time))
                        if os.path.exists(second_model_path):
                            os.remove(second_model_path)
                        save_Multimodal_model(multimodal_model,hyp_params, current_time)
                        best_valid_f1 = macro_f1
                        best_model_time = current_time
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        counter = 0
                    else:
                        counter += 1
                        if counter >= patience:
                            print("Validation loss has not descended for {} epochs. Stopping training.".format(patience))
                            break
            print('\n')
        #---------------------------------------------------------------------测试----------------------------------------------------------------------------#

        else:
            trg_test_loader  = self.setup_dataloaders(trg_test_loader, move_to_device=False)  

            ALL_EMOTIONS = ['Neutral', 'Surprise', 'Fear', 'Sad', 'Happy', 'Disgust', 'Anger'] 

            best_multi_model = torch.load(hyp_params.load_best_model_path)
            
            best_multi_model.eval()
            results = []
            with torch.no_grad():
                for i_batch, batch in enumerate(trg_test_loader):
                    if hyp_params.uttORdia == 'utt':
                        batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, \
                                vision_inputs, vision_mask, batchUtt_in_dia_idx = batch
                        logits = best_multi_model(batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, \
                                                        vision_inputs, vision_mask, batchUtt_in_dia_idx)  #batch_size的utterance的结果
                    elif hyp_params.uttORdia == 'dia':
                        batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, \
                                vision_inputs, vision_mask, dia_mask, curr_numUtt_in_dia = batch
                        logits = best_multi_model(batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, \
                                                        vision_inputs, vision_mask, dia_mask, curr_numUtt_in_dia)  #batch_size的dialogue下的utterance的结果
                    results.append(logits)
            test_preds = torch.cat(results, dim=0).cpu().detach().numpy()
            predicted_label = []
            for i in range(test_preds.shape[0]):
                predicted_label.append(np.argmax(test_preds[i,:],axis=0) ) 

            predicted_label_new = []
            for index in range(len(predicted_label)):
                label_ix = int(predicted_label[index])
                predicted_label_new.append(ALL_EMOTIONS[label_ix])
            load_project_path = os.path.abspath(os.path.dirname(__file__))
            import pandas as pd

            df = pd.read_csv(os.path.join(load_project_path,'nustm_submission_empty.csv'))
            for i in range(len(predicted_label_new)):
                df.iloc[i, 1] = predicted_label_new[i]

            df.to_csv(os.path.join(load_project_path,'nustm_submission.csv'))


