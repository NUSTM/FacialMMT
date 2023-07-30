import torch
import torch.nn as nn
from modules.Transformer import  MELDTransEncoder, AdditiveAttention
from modules.CrossmodalTransformer import CrossModalTransformerEncoder
from transformers import RobertaModel
from transformers import BertModel
from modules.SwinTransformer.backbone_def import BackboneFactory
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

'''SwinTransformer'''
class SwinForAffwildClassification(nn.Module):

    def __init__(self, args):
        super(SwinForAffwildClassification,self).__init__()
        self.num_labels = args.num_labels
        self.swin = BackboneFactory(args.backbone_type, args.backbone_conf_file).get_backbone() #加载Swin-tiny模型
        # Classifier head
        self.linear = nn.Linear(512, 64)
        self.nonlinear = nn.ReLU()
        self.classifier = nn.Linear(64, args.num_labels)
        self.tau = args.tau  #temperature parameter

    def forward(self, images_feature=None, is_trg_task=None, labels=None, criterion=None):
        outputs = self.swin(images_feature) 
        outputs = self.linear(outputs)
        outputs = self.nonlinear(outputs)
        logits = self.classifier(outputs) 
        if is_trg_task:
            logits = F.gumbel_softmax(logits, self.tau)  
        if labels is not None:
            loss = criterion(logits, labels) #cross entropy loss
            return loss
        else:
            return logits


'''multimodal model'''
class MultiModalTransformerForClassification(nn.Module):

    def __init__(self, config):
        super(MultiModalTransformerForClassification,self).__init__()
        self.choice_modality = config.choice_modality
        self.num_labels = config.num_labels
        self.get_text_utt_max_lens = config.get_text_utt_max_lens
        self.hidden_size = config.hidden_size
        if config.pretrainedtextmodel_path.split('/')[-1] == 'roberta-large':
            self.text_pretrained_model = 'roberta'
        else:
            self.text_pretrained_model = 'bert'

        self.audio_emb_dim = config.audio_featExtr_dim
        self.audio_utt_Transformernum = config.audio_utt_Transformernum
        self.get_audio_utt_max_lens = config.get_audio_utt_max_lens
        
        #crossmodal transformer
        self.crossmodal_num_heads_TA = config.crossmodal_num_heads_TA
        self.crossmodal_layers_TA = config.crossmodal_layers_TA
        self.crossmodal_attn_dropout_TA = config.crossmodal_attn_dropout_TA
        
        self.crossmodal_num_heads_TA_V = config.crossmodal_num_heads_TA_V
        self.crossmodal_layers_TA_V = config.crossmodal_layers_TA_V
        self.crossmodal_attn_dropout_TA_V = config.crossmodal_attn_dropout_TA_V
        
        self.vision_emb_dim = config.vision_featExtr_dim + config.num_labels
        self.vision_utt_Transformernum = config.vision_utt_Transformernum
        self.get_vision_utt_max_lens = config.get_vision_utt_max_lens
        
        '''Textual modality through RoBERTa or BERT'''
        if self.text_pretrained_model == 'roberta':
            self.roberta = RobertaModel.from_pretrained(config.pretrainedtextmodel_path)
            self.text_linear = nn.Linear(self.roberta.config.hidden_size, self.hidden_size)
        else:
            self.bert = BertModel.from_pretrained(config.pretrainedtextmodel_path)
            self.text_linear = nn.Linear(self.bert.config.hidden_size, self.hidden_size)

        self.audio_linear = nn.Linear(self.audio_emb_dim, self.hidden_size)
        self.audio_utt_transformer = MELDTransEncoder(config, self.audio_utt_Transformernum,self.get_audio_utt_max_lens, self.hidden_size)  #执行self-attention transformer
    

        self.vision_linear = nn.Linear(self.vision_emb_dim, self.hidden_size)
        self.vision_utt_transformer = MELDTransEncoder(config, self.vision_utt_Transformernum,self.get_vision_utt_max_lens, self.hidden_size)  #执行self-attention transformer

        self.attention = AdditiveAttention(self.hidden_size, self.hidden_size)

        self.CrossModalTrans_TA = CrossModalTransformerEncoder(self.hidden_size, self.crossmodal_num_heads_TA, self.crossmodal_layers_TA, self.crossmodal_attn_dropout_TA)

        self.CrossModalTrans_TA_V = CrossModalTransformerEncoder(self.hidden_size, self.crossmodal_num_heads_TA_V, self.crossmodal_layers_TA_V, self.crossmodal_attn_dropout_TA_V)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, batch_text_input_ids=None, batch_text_input_mask=None, batch_text_sep_mask=None, 
                        audio_inputs=None, audio_mask=None, vision_inputs=None, new_vision_mask=None, 
                        batchUtt_in_dia_idx=None):

        if self.text_pretrained_model == 'roberta':
            #<s>utt_1</s></s>utt_2</s></s>utt_3</s>...
            outputs = self.roberta(batch_text_input_ids, batch_text_input_mask)
        else:
            #[CLS]utt_1[SEP]utt_2[SEP]utt_3[SEP]...
            outputs = self.bert(batch_text_input_ids, batch_text_input_mask)

        text_pretrained_model_out = outputs[0]  # (num_dia, max_sequence_len, 1024)
        text_utt_linear = self.text_linear(text_pretrained_model_out) #(num_dia, max_sequence_len, hidden_size)
        
        '''
        Extract word-level textual representations for each utterance.
        '''
        utt_batch_size = vision_inputs.shape[0]

        batch_text_feat_update = torch.zeros((utt_batch_size, self.get_text_utt_max_lens, text_utt_linear.shape[-1])).cuda()  #batch_size, max_utt_len, hidden_size
        batch_text_sep_mask_update = torch.zeros((utt_batch_size, self.get_text_utt_max_lens)).cuda() #batch_size, max_utt_len

        for i in range(utt_batch_size):
            curr_utt_in_dia_idx = batchUtt_in_dia_idx[i] #the position of the target utterance in the current dialogue.
            curr_dia_mask = batch_text_sep_mask[i]
            each_utt_index = []

            for index, value in enumerate(list(curr_dia_mask)): 
                if value == 1:  
                    each_utt_index.append(index)  #record the index position of the first </s> or [SEP] token for each utterance.
                    if curr_utt_in_dia_idx  == 0:  
                        '''the current utterance is at the 0th position in the dialogue.'''
                        curr_utt_len = index-1   #remove the starting <s> and ending </s> tokens, or remove the starting [CLS] and ending [SEP] tokens.
                        if curr_utt_len > self.get_text_utt_max_lens:
                            curr_utt_len = self.get_text_utt_max_lens
                        batch_text_feat_update[i][:curr_utt_len] = text_utt_linear[i][1:curr_utt_len+1] #从<s>或者[CLS]之后开始
                        batch_text_sep_mask_update[i][:curr_utt_len] = 1
                        break
                    elif curr_utt_in_dia_idx >0 and curr_utt_in_dia_idx + 1 == len(each_utt_index): 
                        curr_ut_id = len(each_utt_index) -1 #1 
                        pre_ut_id = len(each_utt_index) - 2 #0
                        curr_ut = each_utt_index[curr_ut_id]
                        pre_ut = each_utt_index[pre_ut_id]
                        if self.text_pretrained_model == 'roberta':
                            curr_utt_len = curr_ut - pre_ut - 2  #remove </s> and <s>
                        else:
                            curr_utt_len = curr_ut - pre_ut - 1 #remove [SEP]

                        if curr_utt_len > self.get_text_utt_max_lens:
                            curr_utt_len = self.get_text_utt_max_lens
                        if self.text_pretrained_model == 'roberta':
                            batch_text_feat_update[i][:curr_utt_len] = text_utt_linear[i][pre_ut+2:pre_ut+2+curr_utt_len]  #从</s><s>之后开始
                        else:
                            batch_text_feat_update[i][:curr_utt_len] = text_utt_linear[i][pre_ut+1:pre_ut+1+curr_utt_len]  #从[SEP]之后开始
                        batch_text_sep_mask_update[i][:curr_utt_len] = 1
                        break
        #for memory 
        del text_utt_linear, batch_text_sep_mask

        '''audio modality'''
        #Input dim: (batch_size, Max_utt_len, pretrained_wav2vec_dim)
        audio_extended_utt_mask = audio_mask.unsqueeze(1).unsqueeze(2)
        audio_extended_utt_mask = (1.0 - audio_extended_utt_mask) * -10000.0
        audio_emb_linear = self.audio_linear(audio_inputs) 
        audio_utt_trans = self.audio_utt_transformer(audio_emb_linear, audio_extended_utt_mask)    #(batch_size, utt_max_lens, self.hidden_size)
        
        '''visual modality'''
        #Input dim: (batch_size, Max_utt_len, pretrained_IncepRes_dim + 7)
        vision_extended_utt_mask = new_vision_mask.unsqueeze(1).unsqueeze(2)
        vision_extended_utt_mask = (1.0 - vision_extended_utt_mask) * -10000.0  
        vision_emb_linear = self.vision_linear(vision_inputs) 
        vision_utt_trans = self.vision_utt_transformer(vision_emb_linear, vision_extended_utt_mask) #(batch_size, utt_max_lens, self.hidden_size)

        # cross-modality
        batch_text_feat_update = batch_text_feat_update.transpose(0,1)
        audio_utt_trans = audio_utt_trans.transpose(0,1)
        text_crossAudio_att = self.CrossModalTrans_TA(batch_text_feat_update, audio_utt_trans, audio_utt_trans) #(max_textUtt_len,  batch_size, self.hidden_size)
        audio_crossText_att = self.CrossModalTrans_TA(audio_utt_trans, batch_text_feat_update, batch_text_feat_update) #(max_audioUtt_len, batch_size, self.hidden_size)
        textaudio_cross_feat = torch.concat((text_crossAudio_att, audio_crossText_att),dim=0) #(max_textUtt_len + max_audioUtt_len, batch_size, self.hidden_size)

        vision_utt_trans = vision_utt_trans.transpose(0,1)
        vision_crossTeAud_att = self.CrossModalTrans_TA_V(vision_utt_trans, textaudio_cross_feat, textaudio_cross_feat) #(max_visionUtt_len, batch_size, self.hidden_size)
        texaudi_crossVis_att = self.CrossModalTrans_TA_V(textaudio_cross_feat, vision_utt_trans, vision_utt_trans) #(max_textUtt_len + max_audioUtt_len, batch_size, self.hidden_size)
        final_utt_feat = torch.concat((texaudi_crossVis_att, vision_crossTeAud_att),dim=0) #(max_textUtt_len + max_audioUtt_len + max_visionUtt_len, batch_size, self.hidden_size)
        final_utt_feat = final_utt_feat.transpose(0,1) 
        T_A_utt_mask = torch.concat((batch_text_sep_mask_update, audio_mask),dim=1)
        final_utt_mask = torch.concat((T_A_utt_mask, new_vision_mask),dim=1)

        multimodal_out, _ = self.attention(final_utt_feat, final_utt_mask) #（batch_size, self.hidden_size）

        # classify
        multimodal_output = self.dropout(multimodal_out)  
        logits = self.classifier(multimodal_output)
        return logits


'''vision model'''
class meld_utt_transformer(nn.Module):
    def __init__(self, args):
        super(meld_utt_transformer, self).__init__()

        num_labels = args.num_labels
        self.modality_origin_emb = args.vision_featExtr_dim
        self.modality_utt_Transformernum = args.vision_utt_Transformernum
        self.get_utt_max_lens = args.get_vision_utt_max_lens   

        self.hidden_size = args.hidden_size
        self.hidden_dropout_prob = args.hidden_dropout_prob   
        self.modality_linear = nn.Linear(self.modality_origin_emb, self.hidden_size)
        self.utt_transformer = MELDTransEncoder(args, self.modality_utt_Transformernum, self.get_utt_max_lens, self.hidden_size) 
        self.attention = AdditiveAttention(self.hidden_size, self.hidden_size)  
        self.mm_dropout = nn.Dropout(self.hidden_dropout_prob)  
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, inputs=None, utt_mask=None):
        '''
        input_shape: (batch_size, utt_max_lens, featureExtractor_dim)  
        utt_mask_shape: (batch_size, utt_max_lens)
        '''
        extended_utt_mask = utt_mask.unsqueeze(1).unsqueeze(2)
        extended_utt_mask = extended_utt_mask.to(dtype=next(self.parameters()).dtype)
        extended_utt_mask = (1.0 - extended_utt_mask) * -10000.0  
        modality_emb_linear = self.modality_linear(inputs)
        modality_utt_trans = self.utt_transformer(modality_emb_linear, extended_utt_mask)  #(batch_size, utt_max_lens, self.hidden_size)
        utt_trans_attention, _ = self.attention(modality_utt_trans, utt_mask) #(batch_size, self.hidden_size)
        outputs = self.mm_dropout(utt_trans_attention)  
        logits = self.classifier(outputs)  #(batch_size, 7)

        return logits
