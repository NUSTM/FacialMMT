import torch
import torch.nn as nn
from modules.Transformer import  MELDTransEncoder, AdditiveAttention
from modules.CrossmodalTransformer import CrossModalTransformerEncoder
from transformers import BertModel  #中文版的roberta用的依然是这个包
import torch.nn.functional as F



class MultiModalTransformerForClassification_utt_level(nn.Module):

    def __init__(self, config):
        super(MultiModalTransformerForClassification_utt_level,self).__init__()
        self.choice_modality = config.choice_modality
        self.num_labels = config.num_labels
        self.get_text_utt_max_lens = config.get_text_utt_max_lens
        self.hidden_size = config.hidden_size
        self.modalityFuse = config.modalityFuse

        self.audio_emb_dim = config.audio_featExtr_dim
        self.audio_utt_Transformernum = config.audio_utt_Transformernum
        self.get_audio_utt_max_lens = config.get_audio_utt_max_lens

        #crossmodal transformer
        if config.modalityFuse == 'crossmodal':
            if config.choice_modality in ('T+A','T+A+V') :
                self.crossmodal_num_heads_TA = config.crossmodal_num_heads_TA
                self.crossmodal_layers_TA = config.crossmodal_layers_TA
                self.crossmodal_attn_dropout_TA = config.crossmodal_attn_dropout_TA
            
            if config.choice_modality == 'T+A+V':
                self.crossmodal_num_heads_TA_V = config.crossmodal_num_heads_TA_V
                self.crossmodal_layers_TA_V = config.crossmodal_layers_TA_V
                self.crossmodal_attn_dropout_TA_V = config.crossmodal_attn_dropout_TA_V
            
            if config.choice_modality == 'T+V':
                self.crossmodal_num_heads_TV = config.crossmodal_num_heads_TV
                self.crossmodal_layers_TV = config.crossmodal_layers_TV
                self.crossmodal_attn_dropout_TV = config.crossmodal_attn_dropout_TV

            self.crossmodal_gelu_dropout = config.crossmodal_gelu_dropout
            self.crossmodal_res_dropout = config.crossmodal_res_dropout
            self.crossmodal_embed_dropout = config.crossmodal_embed_dropout
            self.crossmodal_attn_mask = config.crossmodal_attn_mask

        self.vision_emb_dim = config.vision_featExtr_dim
        self.vision_utt_Transformernum = config.vision_utt_Transformernum
        self.get_vision_utt_max_lens = config.get_vision_utt_max_lens
        
        '''文本模态过中文版的roberta'''
        self.roberta = BertModel.from_pretrained(config.pretrainedtextmodel_path) 
        self.text_linear = nn.Linear(self.roberta.config.hidden_size, self.hidden_size)

        '''语音模态过一层全连接+自注意力transformer+聚合操作,拿到utterance级别表示'''
        if self.choice_modality in ('T+A','T+A+V'):
            self.audio_linear = nn.Linear(self.audio_emb_dim, self.hidden_size)
            self.audio_utt_transformer = MELDTransEncoder(config, self.audio_utt_Transformernum,self.get_audio_utt_max_lens, self.hidden_size)  #执行self-attention transformer
        
        '''视觉模态过一层全连接+自注意力transformer+聚合操作,拿到utterance级别表示'''
        if self.choice_modality in ('T+V', 'T+A+V'):
            self.vision_linear = nn.Linear(self.vision_emb_dim, self.hidden_size)
            self.vision_utt_transformer = MELDTransEncoder(config, self.vision_utt_Transformernum,self.get_vision_utt_max_lens, self.hidden_size)  #执行self-attention transformer

        # #进行词级别词级注意力机制加权聚合操作,拿到utterance级别的表示
        self.attention = AdditiveAttention(self.hidden_size, self.hidden_size)

        if self.modalityFuse == 'concat':
            '''三模态concat过一层全连接'''
            if config.choice_modality == 'T+A+V':
                self.multimodal_linear = nn.Linear(self.hidden_size * 3, self.hidden_size)
            elif config.choice_modality in ('T+A','T+V'):
                self.multimodal_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

        if self.modalityFuse == 'crossmodal':
            '''
            执行crossmodal transformer, 我们设置共4个CMT, 首先文本模态和语音模态进行跨模态交互, 之后拿到拼接结果, 再和视觉模态进行交互.
            '''
            if self.choice_modality in ('T+A', 'T+A+V'):
                self.CrossModalTrans_TA = CrossModalTransformerEncoder(self.hidden_size, self.crossmodal_num_heads_TA, self.crossmodal_layers_TA, self.crossmodal_attn_dropout_TA,\
                                                self.crossmodal_gelu_dropout, self.crossmodal_res_dropout, self.crossmodal_embed_dropout, self.crossmodal_attn_mask)
            
            if self.choice_modality == 'T+V':
                self.CrossModalTrans_TV = CrossModalTransformerEncoder(self.hidden_size, self.crossmodal_num_heads_TV, self.crossmodal_layers_TV, self.crossmodal_attn_dropout_TV,\
                                                self.crossmodal_gelu_dropout, self.crossmodal_res_dropout, self.crossmodal_embed_dropout, self.crossmodal_attn_mask)

            if self.choice_modality == 'T+A+V':
                self.CrossModalTrans_TA_V = CrossModalTransformerEncoder(self.hidden_size, self.crossmodal_num_heads_TA_V, self.crossmodal_layers_TA_V, self.crossmodal_attn_dropout_TA_V,\
                                                self.crossmodal_gelu_dropout, self.crossmodal_res_dropout, self.crossmodal_embed_dropout, self.crossmodal_attn_mask)
        '''过dropout+全连接'''
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, batch_text_input_ids=None, batch_text_input_mask=None, batch_text_sep_mask=None, 
                        audio_inputs=None, audio_mask=None, vision_inputs=None, new_vision_mask=None, 
                        batchUtt_in_dia_idx=None):

        '''文本模态先经过Roberta-chinese'''
        #文本格式是[CLS]utt_1[SEP]utt_2[SEP]utt_3[SEP]...
        outputs = self.roberta(batch_text_input_ids, batch_text_input_mask)

        text_pretrained_model_out = outputs[0]  # (num_dia, max_sequence_len, 1024)
        text_utt_linear = self.text_linear(text_pretrained_model_out) #(num_dia, max_sequence_len, hidden_size)
        '''
        这里我们每次送进去是当前utterance的整个dialogue, 但是只取出该utterance位置的表示
        (2022.11.11)
        需要重新设计batch_text_sep_mask, 取出每个utterance的词级别表示
        '''
        if audio_inputs != None:
            utt_batch_size = audio_inputs.shape[0]
        if vision_inputs != None:
            utt_batch_size = vision_inputs.shape[0]

        batch_text_feat_update = torch.zeros((utt_batch_size, self.get_text_utt_max_lens, text_utt_linear.shape[-1])).cuda()  #batch_size, max_utt_len, hidden_size
        batch_text_sep_mask_update = torch.zeros((utt_batch_size, self.get_text_utt_max_lens)).cuda() #batch_size, max_utt_len

        '''先找该utt在当前dia中的位置(第几个1), 之后取出当前1到上个1中的所有的[0]'''
        for i in range(utt_batch_size):
            curr_utt_in_dia_idx = batchUtt_in_dia_idx[i] #目标utterance在当前dialgue中的位置
            curr_dia_mask = batch_text_sep_mask[i]
            each_utt_index = []

            for index, value in enumerate(list(curr_dia_mask)): 
                if value == 1:  
                    each_utt_index.append(index)  #记录每个utterance的第一个</s>或者[SEP]下标位置 
                    if curr_utt_in_dia_idx  == 0:  
                        '''当前utterance位于dialogue的第0个位置'''
                        curr_utt_len = index-1   #去掉开头的<s>和结束符</s>，或者是去掉开头的[CLS]和结束符[SEP]
                        if curr_utt_len > self.get_text_utt_max_lens:
                            curr_utt_len = self.get_text_utt_max_lens
                        batch_text_feat_update[i][:curr_utt_len] = text_utt_linear[i][1:curr_utt_len+1] #从<s>或者[CLS]之后开始
                        batch_text_sep_mask_update[i][:curr_utt_len] = 1
                        break
                    elif curr_utt_in_dia_idx >0 and curr_utt_in_dia_idx + 1 == len(each_utt_index): 
                        '''当前utterance位于dialogue的其他位置'''
                        curr_ut_id = len(each_utt_index) -1 #1 
                        pre_ut_id = len(each_utt_index) - 2 #0
                        curr_ut = each_utt_index[curr_ut_id]
                        pre_ut = each_utt_index[pre_ut_id]
                        # if self.text_pretrained_model == 'roberta':
                        #     curr_utt_len = curr_ut - pre_ut - 2  #去掉</s>和<s>
                        # else:
                        curr_utt_len = curr_ut - pre_ut - 1 #去掉[SEP]

                        if curr_utt_len > self.get_text_utt_max_lens:
                            curr_utt_len = self.get_text_utt_max_lens
                        # if self.text_pretrained_model == 'roberta':
                        #     batch_text_feat_update[i][:curr_utt_len] = text_utt_linear[i][pre_ut+2:pre_ut+2+curr_utt_len]  #从</s><s>之后开始
                        # else:
                        batch_text_feat_update[i][:curr_utt_len] = text_utt_linear[i][pre_ut+1:pre_ut+1+curr_utt_len]  #从[SEP]之后开始
                        batch_text_sep_mask_update[i][:curr_utt_len] = 1
                        break
        #for memory 
        del text_utt_linear, batch_text_sep_mask

        '''语音模态过线性层+self-attention transformer'''
        if audio_inputs != None: 
            #输入进来的维度是: (batch_size, Max_utt_len, pretrained_wav2vec_dim)
            audio_extended_utt_mask = audio_mask.unsqueeze(1).unsqueeze(2)
            audio_extended_utt_mask = (1.0 - audio_extended_utt_mask) * -10000.0  #实际的位置变为了0，padding的部分变为了-10000  和真实的一加, 实际位置没有影响, padding部分无穷大, 经过softmax都变为0
            audio_emb_linear = self.audio_linear(audio_inputs) 
            audio_utt_trans = self.audio_utt_transformer(audio_emb_linear, audio_extended_utt_mask)  #utterance内部交互  #(batch_size, utt_max_lens, self.hidden_size)
        
        '''视觉模态过线性层+self-attention transformer'''
        if vision_inputs != None:
            #输入的维度是: (batch_size, Max_utt_len, pretrained_IncepRes_dim + 7)
            vision_extended_utt_mask = new_vision_mask.unsqueeze(1).unsqueeze(2)
            vision_extended_utt_mask = (1.0 - vision_extended_utt_mask) * -10000.0  
            vision_emb_linear = self.vision_linear(vision_inputs) 
            vision_utt_trans = self.vision_utt_transformer(vision_emb_linear, vision_extended_utt_mask)  #utterance内部交互  #(batch_size, utt_max_lens, self.hidden_size)

        if self.modalityFuse == 'crossmodal':
            #先执行文本模态和语音模态的跨模态
            '''
            这里缺失文本模态的单独编码表示,即(batch_size, MAX_UTT_LEN, feat_dim),
            '''
            batch_text_feat_update = batch_text_feat_update.transpose(0,1)
            
            if self.choice_modality in ('T+A', 'T+A+V'): 
                audio_utt_trans = audio_utt_trans.transpose(0,1)

                text_crossAudio_att = self.CrossModalTrans_TA(batch_text_feat_update, audio_utt_trans, audio_utt_trans) #(max_textUtt_len,  batch_size, self.hidden_size)
                
                audio_crossText_att = self.CrossModalTrans_TA(audio_utt_trans, batch_text_feat_update, batch_text_feat_update) #(max_audioUtt_len, batch_size, self.hidden_size)

                textaudio_cross_feat = torch.concat((text_crossAudio_att, audio_crossText_att),dim=0) #(max_textUtt_len + max_audioUtt_len, batch_size, self.hidden_size)

                if self.choice_modality == 'T+A': 
                    final_utt_feat = textaudio_cross_feat.transpose(0, 1)
                    final_utt_mask = torch.concat((batch_text_sep_mask_update, audio_mask),dim=1)

                #再执行(文本语音)和视觉的跨模态
                elif self.choice_modality == 'T+A+V': 
                    vision_utt_trans = vision_utt_trans.transpose(0,1)
                    vision_crossTeAud_att = self.CrossModalTrans_TA_V(vision_utt_trans, textaudio_cross_feat, textaudio_cross_feat) #(max_visionUtt_len, batch_size, self.hidden_size)
                    texaudi_crossVis_att = self.CrossModalTrans_TA_V(textaudio_cross_feat, vision_utt_trans, vision_utt_trans) #(max_textUtt_len + max_audioUtt_len, batch_size, self.hidden_size)
                    final_utt_feat = torch.concat((texaudi_crossVis_att, vision_crossTeAud_att),dim=0) #(max_textUtt_len + max_audioUtt_len + max_visionUtt_len, batch_size, self.hidden_size)
                    final_utt_feat = final_utt_feat.transpose(0,1) 
                    T_A_utt_mask = torch.concat((batch_text_sep_mask_update, audio_mask),dim=1)
                    final_utt_mask = torch.concat((T_A_utt_mask, new_vision_mask),dim=1)

            if self.choice_modality == 'T+V': 
                vision_utt_trans = vision_utt_trans.transpose(0,1)
                text_crossVision_att = self.CrossModalTrans_TV(batch_text_feat_update, vision_utt_trans, vision_utt_trans) #(max_textUtt_len,  batch_size, self.hidden_size)
                vision_crossText_att = self.CrossModalTrans_TV(vision_utt_trans, batch_text_feat_update, batch_text_feat_update) #(max_visionUtt_len, batch_size, self.hidden_size)
                textvision_cross_feat = torch.concat((text_crossVision_att, vision_crossText_att),dim=0) #(max_textUtt_len + max_visionUtt_len, batch_size, self.hidden_size)
                final_utt_feat = textvision_cross_feat.transpose(0, 1) #(batch_size, max_textUtt_len + max_visionUtt_len, self.hidden_size)
                final_utt_mask = torch.concat((batch_text_sep_mask_update, new_vision_mask),dim=1)

            multimodal_out, _ = self.attention(final_utt_feat, final_utt_mask) #（batch_size, max_textUtt_len + max_visionUtt_len+ max_AudioUtt_len,self.hidden_size）

        if self.modalityFuse == 'concat':
            # text_utt_feat = batch_text_feat_update.masked_select(batch_text_sep_mask_update.unsqueeze(2).bool()).reshape(-1, self.hidden_size) #(num_valid_utt_of_dia, hidden_size)
            text_utt_feat, _ = self.attention(batch_text_feat_update, batch_text_sep_mask_update) #(batch_size, hidden_size) 
            if self.choice_modality in ('T+A', 'T+A+V'):
                audio_utt_out, _ = self.attention(audio_utt_trans, audio_mask) #（batch_size, self.hidden_size）
                if self.choice_modality == 'T+A+V':
                    vision_utt_out, _ = self.attention(vision_utt_trans, new_vision_mask) #(batch_size, hidden_size)
                    multimodal_input = torch.cat((text_utt_feat, audio_utt_out, vision_utt_out),dim=-1)
                else:
                    multimodal_input = torch.cat((text_utt_feat, audio_utt_out),dim=-1)
            multimodal_out = self.multimodal_linear(multimodal_input)
        '''dropout层和分类器'''
        multimodal_output = self.dropout(multimodal_out)  
        logits = self.classifier(multimodal_output)
        return logits


class MultiModalTransformerForClassification_dia_level(nn.Module):

    def __init__(self, config):
        super(MultiModalTransformerForClassification_dia_level,self).__init__()
        self.choice_modality = config.choice_modality
        self.num_labels = config.num_labels
        # self.get_text_utt_max_lens = config.get_text_utt_max_lens
        self.hidden_size = config.hidden_size
        self.modalityFuse = config.modalityFuse

        self.audio_emb_dim = config.audio_featExtr_dim
        self.audio_utt_Transformernum = config.audio_utt_Transformernum
        self.get_audio_utt_max_lens = config.get_audio_utt_max_lens

        #crossmodal transformer
        if config.modalityFuse == 'crossmodal':
            self.crossmodal_num_heads_TA = config.crossmodal_num_heads_TA
            self.crossmodal_layers_TA = config.crossmodal_layers_TA
            self.crossmodal_attn_dropout_TA = config.crossmodal_attn_dropout_TA
        
            self.crossmodal_num_heads_TA_V = config.crossmodal_num_heads_TA_V
            self.crossmodal_layers_TA_V = config.crossmodal_layers_TA_V
            self.crossmodal_attn_dropout_TA_V = config.crossmodal_attn_dropout_TA_V
    
            self.crossmodal_gelu_dropout = config.crossmodal_gelu_dropout
            self.crossmodal_res_dropout = config.crossmodal_res_dropout
            self.crossmodal_embed_dropout = config.crossmodal_embed_dropout
            self.crossmodal_attn_mask = config.crossmodal_attn_mask

        self.vision_emb_dim = config.vision_featExtr_dim
        self.vision_utt_Transformernum = config.vision_utt_Transformernum
        self.get_vision_utt_max_lens = config.get_vision_utt_max_lens

        '''文本模态过中文版的roberta'''
        self.roberta = BertModel.from_pretrained(config.pretrainedtextmodel_path) 
        self.text_linear = nn.Linear(self.roberta.config.hidden_size, self.hidden_size)

        '''语音模态过一层全连接+自注意力transformer+聚合操作,拿到utterance级别表示'''
        if self.choice_modality in ('T+A','T+A+V'):
            self.audio_linear = nn.Linear(self.audio_emb_dim, self.hidden_size)
            self.audio_utt_transformer = MELDTransEncoder(config, self.audio_utt_Transformernum,self.get_audio_utt_max_lens, self.hidden_size)  #执行self-attention transformer
        
        '''视觉模态过一层全连接+自注意力transformer+聚合操作,拿到utterance级别表示'''
        if self.choice_modality in ('T+V', 'T+A+V'):
            self.vision_linear = nn.Linear(self.vision_emb_dim, self.hidden_size)
            self.vision_utt_transformer = MELDTransEncoder(config, self.vision_utt_Transformernum,self.get_vision_utt_max_lens, self.hidden_size)  #执行self-attention transformer

        # #进行词级别词级注意力机制加权聚合操作,拿到utterance级别的表示
        self.attention_pooling = AdditiveAttention(self.hidden_size, self.hidden_size)

        if self.modalityFuse == 'crossmodal':
            '''
            执行crossmodal transformer, 我们设置共4个CMT, 首先文本模态和语音模态进行跨模态交互, 之后拿到拼接结果, 再和视觉模态进行交互.
            '''

            self.CrossModalTrans_TA = CrossModalTransformerEncoder(self.hidden_size, self.crossmodal_num_heads_TA, self.crossmodal_layers_TA, self.crossmodal_attn_dropout_TA,\
                                                self.crossmodal_gelu_dropout, self.crossmodal_res_dropout, self.crossmodal_embed_dropout, self.crossmodal_attn_mask)
            
            self.CrossModalTrans_TA_V = CrossModalTransformerEncoder(self.hidden_size, self.crossmodal_num_heads_TA_V, self.crossmodal_layers_TA_V, self.crossmodal_attn_dropout_TA_V,\
                                                self.crossmodal_gelu_dropout, self.crossmodal_res_dropout, self.crossmodal_embed_dropout, self.crossmodal_attn_mask)
        
        self.multimodal_linear = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.multimodal_linear2 = nn.Linear(self.hidden_size * 2, self.hidden_size)

        '''过dropout+全连接'''
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, batch_text_input_ids=None, batch_text_input_mask=None, batch_text_sep_mask=None, 
                        audio_inputs=None, audio_mask=None, vision_inputs=None, vision_mask=None, 
                        dia_mask=None, curr_numUtt_in_dia=None):

        '''语音模态过线性层+self-transformer+聚合操作拿到utterance级别表示'''
        #输入进来的维度是: (batch_size, Max_dia_len, Max_utt_len, pretrained_wav2vec_dim)
        batch_size, MAX_DIA_LEN, audio_max_utt_len, featureExtractor_dim = audio_inputs.size()
        
        audio_mask = audio_mask.reshape(-1, audio_max_utt_len) #(batch_size * max_dia_len, max_utt_len)
        
        audio_extended_utt_mask = audio_mask.unsqueeze(1).unsqueeze(2)
        audio_extended_utt_mask = audio_extended_utt_mask.to(dtype=next(self.parameters()).dtype)
        audio_extended_utt_mask = (1.0 - audio_extended_utt_mask) * -10000.0  #实际的位置变为了0，padding的部分变为了-10000  和真实的一加, 实际位置没有影响, padding部分无穷大, 经过softmax都变为0
        
        #reshape: 得到utterance级别的形状: (batch_size*max_dialogue_len, max_utt_len, feat_dim)
        audio_inputs = audio_inputs.reshape(-1, audio_max_utt_len, featureExtractor_dim)  #  
        audio_emb_linear = self.audio_linear(audio_inputs) 
        audio_utt_trans = self.audio_utt_transformer(audio_emb_linear, audio_extended_utt_mask)  #utterance内部交互  #(batch_size*audio_max_dia_len, utt_max_lens, self.hidden_size)
        
        audio_utt_trans_out, _ = self.attention_pooling(audio_utt_trans, audio_mask) #（batch_size*audio_max_dia_len, 1, self.hidden_size）
        audio_utt_trans = audio_utt_trans_out.squeeze(1).reshape(batch_size, MAX_DIA_LEN, -1)  #(batch_size, max_dia_len, self.hidden_size)

        '''文本模态先经过Roberta-chinese, 拿到utterance级别表示(batch_size, max_dia_len, hidden_size)'''
        #文本格式是[CLS]utt_1[SEP]utt_2[SEP]utt_3[SEP]...
        outputs = self.roberta(batch_text_input_ids, batch_text_input_mask)
        text_pretrained_model_out = outputs[0]  # (batch_size, max_sequence_len(512), 1024)
        text_utt_linear = self.text_linear(text_pretrained_model_out) #(batch_size, max_sequence_len(512), hidden_size)
        text_utt_feat = text_utt_linear.masked_select(batch_text_sep_mask.unsqueeze(2).bool()).reshape(-1, self.hidden_size)  #(num_valid_utt, hidden_size)
        #通过每个dialogue包含多少个utterance,将(num_valid_utt, hidden_size)转为(batch_size, max_dia_len, hidden_size)
        text_utt_feat_update = torch.zeros((batch_size, MAX_DIA_LEN, self.hidden_size)).cuda() #(batch_size, max_dia_len, hidden_size)
        for i in range(batch_size):
            curr_utt_index_start = 0 if i == 0 else curr_utt_index_start + curr_numUtt_in_dia[i-1]
            curr_utt_len = curr_numUtt_in_dia[i]
            text_utt_feat_update[i, :curr_numUtt_in_dia[i], :] = text_utt_feat[curr_utt_index_start:curr_utt_len+curr_utt_index_start, :]

        '''视觉模态过线性层+self-transformer+聚合操作拿到utterance级别表示'''
        _, MAX_DIA_LEN, vision_max_utt_len, featureExtractor_emo_dim = vision_inputs.size()
        #输入的维度是: (batch_size, Max_dia_len, Max_utt_len, pretrained_IncepRes_dim)
        vision_mask = vision_mask.reshape(-1, vision_max_utt_len) #(batch_size * max_dia_len, max_utt_len)
        vision_extended_utt_mask = vision_mask.unsqueeze(1).unsqueeze(2)
        vision_extended_utt_mask = vision_extended_utt_mask.to(dtype=next(self.parameters()).dtype)
        vision_extended_utt_mask = (1.0 - vision_extended_utt_mask) * -10000.0  #实际的位置变为了0，padding的部分变为了-10000  和真实的一加, 实际位置没有影响, padding部分无穷大, 经过softmax都变为0
        #reshape: 得到utterance级别的形状: (batch_size*dialogue_len, utt_len, feat_dim+7)
        vision_inputs = vision_inputs.reshape(-1, vision_max_utt_len, featureExtractor_emo_dim)  #  
        vision_emb_linear = self.vision_linear(vision_inputs)
        vision_utt_trans = self.vision_utt_transformer(vision_emb_linear, vision_extended_utt_mask)  #utterance内部交互  #(batch_size*dialogue_len, utt_max_lens, self.hidden_size)
        vision_utt_trans_out, _ = self.attention_pooling(vision_utt_trans, vision_mask) #
        vision_utt_trans = vision_utt_trans_out.squeeze(1).reshape(batch_size, MAX_DIA_LEN, -1) #(batch_size, max_dia_len, self.hidden_size)

        # # 增加dia_attention的维度
        # extended_dia_mask = dia_mask.unsqueeze(1).unsqueeze(2)
        # extended_dia_mask = extended_dia_mask.to(dtype=next(self.parameters()).dtype)
        # extended_dia_mask = (1.0 - extended_dia_mask) * -10000.0

        if self.modalityFuse == 'crossmodal':
            #先执行文本模态和语音模态的跨模态utterance级别交互
            batch_text_feat_update = text_utt_feat_update.transpose(0,1)
            if self.choice_modality in ('T+A', 'T+A+V'): 
                #先执行文本和语音的跨模态utterance级别交互
                audio_utt_trans = audio_utt_trans.transpose(0,1)
                text_crossAudio_att = self.CrossModalTrans_TA(batch_text_feat_update, audio_utt_trans, audio_utt_trans) #(max_dia_len,  batch_size, self.hidden_size)
                audio_crossText_att = self.CrossModalTrans_TA(audio_utt_trans, batch_text_feat_update, batch_text_feat_update) #(max_dia_len, batch_size, self.hidden_size)
                textaudio_cross_feat = torch.concat((text_crossAudio_att, audio_crossText_att),dim=-1) #(max_dia_len, batch_size, self.hidden_size*2)
                textaudio_cross_feat = self.multimodal_linear2(textaudio_cross_feat)
                #再执行(文本语音)和视觉的跨模态utterance级别交互
                vision_utt_trans = vision_utt_trans.transpose(0,1)
                vision_crossTeAud_att = self.CrossModalTrans_TA_V(vision_utt_trans, textaudio_cross_feat, textaudio_cross_feat) #(max_dia_len, batch_size, self.hidden_size)
                texaudi_crossVis_att = self.CrossModalTrans_TA_V(textaudio_cross_feat, vision_utt_trans, vision_utt_trans) #(max_dia_len, batch_size, self.hidden_size*2)
                final_utt_feat = torch.concat((texaudi_crossVis_att, vision_crossTeAud_att),dim=-1) #(max_dia_len, batch_size, self.hidden_size*2)
                final_utt_feat = final_utt_feat.transpose(0,1)  #(batch_size, max_dia_len, self.hidden_size*2)
                final_utt_feat = self.multimodal_linear2(final_utt_feat) #(batch_size, max_dia_len, self.hidden_size)
                # T_A_utt_mask = torch.concat((dia_mask, dia_mask),dim=1)
                # final_utt_mask = torch.concat((T_A_utt_mask, dia_mask),dim=1)
                multimodal_out = final_utt_feat.masked_select(dia_mask.unsqueeze(2).bool()).reshape(-1, final_utt_feat.shape[-1]) #(num_valid_utt, hidden_size) 
            #(max_textDia_len + max_audioDia_len + max_visionDia_len, batch_size, self.hidden_size)  final_utt_mask的维度应该为()
            # final_utt_mask = final_utt_mask.reshape(-1, MAX_DIA_LEN) #(3,71)
            # multimodal_out, _ = self.attention_pooling(final_utt_feat, final_utt_mask) #（batch_size, self.hidden_size）

        elif self.modalityFuse == 'concat':
            #根据dia_mask, 语音模态输出有效utterance位置上的表示, 输出格式(num_valid_utt, hidden_size)
            audio_utt_feat = audio_utt_trans.masked_select(dia_mask.unsqueeze(2).bool()).reshape(-1, self.hidden_size) #(num_valid_utt, hidden_size)
            vision_utt_feat = vision_utt_trans.masked_select(dia_mask.unsqueeze(2).bool()).reshape(-1, self.hidden_size) #(num_valid_utt, hidden_size)
            multimodal_input = torch.cat((text_utt_feat, audio_utt_feat, vision_utt_feat),dim=-1)
            '''三模态过线性层'''
            multimodal_out = self.multimodal_linear(multimodal_input)
        
        '''dropout层和分类器'''
        multimodal_output = self.dropout(multimodal_out)  
        logits = self.classifier(multimodal_output)
        return logits

