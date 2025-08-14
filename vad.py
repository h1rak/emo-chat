import torch
from transformers import T5Tokenizer
import numpy as np
import argparse, random, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import T5Model, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm, trange,tnrange,tqdm_notebook
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline, T5PreTrainedModel, T5Model
import pandas as pd
import re

class Dense(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.dense = nn.Linear(input_size,hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.out_proj = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class T5EmoMood(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_labels = 7
        self.vad_weight = 0.3
        self.mid_size = 1024
        self.enc = T5Model.from_pretrained("t5-large").encoder
        hid = self.enc.config.d_model  # 1024 (not 512)

        self.mood_dense = Dense(self.mid_size+3,hid,3)
        self.mood_to_hidden = Dense(3,hid,self.mid_size)
        self.mood_to_logit = Dense(3,hid,4)

        self.hidden_resize = Dense(hid,hid,self.mid_size)
        self.personality_to_hidden = nn.Linear(3,self.mid_size)
        self.hidden_to_vad = Dense(hid,hid,3)

        self.classifier = nn.Linear(self.mid_size*3,7)

    def forward(self, ids, mask, uttr_vad, user_emo, personality, init_mood):
        h = self.enc(ids, attention_mask=mask).last_hidden_state.mean(1)
        uttr_vad = self.vad_weight*uttr_vad + (1-self.vad_weight)*user_emo
        delta_mood = torch.cat((uttr_vad,self.hidden_resize(h)),1)
        
        # softmaxの次元を明示的に指定
        response_mood_vad = F.softmax(self.mood_dense(delta_mood)) * personality + init_mood
        response_mood_logits = self.mood_to_logit(response_mood_vad)
        emo_embedding = torch.cat((self.mood_to_hidden(response_mood_vad), h, self.personality_to_hidden(personality)), 1)
        response_emo_logits = self.classifier(emo_embedding)
        return response_mood_vad, response_mood_logits, response_emo_logits

class PredEmo:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.VAD_dict = self.get_vad_dict()
        self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
        self.Emotion_dict = {
                            'anger': [-0.51, 0.59, 0.25],
                            'disgust': [-0.60, 0.35, 0.11],
                            'fear': [-0.62, 0.82, -0.43],
                            'joy': [0.81, 0.51, 0.46],
                            'neutral': [0.0, 0.0, 0.0],
                            'sadness': [-0.63, -0.27, -0.33],
                            'surprise': [0.40, 0.67, -0.13]
                        }
        self.max_len = 128
    
    def personality_to_vad(self,personality):
        O,C,E,A,N = personality[:,0], personality[:,1], personality[:,2], personality[:,3], personality[:,4]
        valence = 0.21 * 10 * E + 0.59 * 10 * A + 0.19 * 10 * N
        arousal = 0.15 * O + 0.30 * 10 * A - 0.57 * 10 * N
        dominance = 0.25 * O + 0.17 * C + 0.60 * 10 * E - 0.32 * 10 * A
        return torch.cat((valence.unsqueeze(-1), arousal.unsqueeze(-1), dominance.unsqueeze(-1)), 1)

    def get_sent_vad_attention(self,VAD_dict, input_id_2, tokenizer, user_emo):
        VAD_scores = []
        
        # トークンデコードの改善
        try:
            if isinstance(input_id_2, list):
                decoded_text = tokenizer.decode(input_id_2, skip_special_tokens=True)
            else:
                decoded_text = tokenizer.decode(input_id_2.tolist(), skip_special_tokens=True)
            
            w_list = re.sub(r'[^\w\s\[\]]','', decoded_text).split()
        except Exception as e:
            print(f"デコードエラー: {e}")
            return [0.0, 0.0, 0.0]
        
        for word in w_list:
            if isinstance(word, str):
                word_lower = word.lower()
                if word_lower in VAD_dict:
                    VAD_scores.append([VAD_dict[word_lower][0], VAD_dict[word_lower][1], VAD_dict[word_lower][2]])
                else:
                    VAD_scores.append([0,0,0])

        if len(VAD_scores) == 0:
            return [0.0, 0.0, 0.0]
            
        VAD_scores = torch.Tensor(VAD_scores)
        user_emo = torch.Tensor(user_emo)
        
        try:
            VAD_scores_weights = torch.inner(VAD_scores, user_emo) 
            VAD_scores_weights = F.softmax(VAD_scores_weights, dim=0) 
        except:
            # フォールバック：均等重み
            VAD_scores_weights = torch.ones(len(VAD_scores)) / len(VAD_scores)
        
        vad_attn = [0,0,0]
        for i in range(len(VAD_scores)):
            vad_attn[0] += VAD_scores[i][0] * VAD_scores_weights[i]
            vad_attn[1] += VAD_scores[i][1] * VAD_scores_weights[i]
            vad_attn[2] += VAD_scores[i][2] * VAD_scores_weights[i]
        
        return [float(vad_attn[0]), float(vad_attn[1]), float(vad_attn[2])]


    def get_vad_dict(self):
        VAD_Lexicons = pd.read_csv('/workspace/santa/PELD-chat/data/source_data/NRC-VAD-Lexicon-v2.1.txt', sep='\t')
        VAD_dict = {}
        for r in VAD_Lexicons.iterrows():
            term = r[1]["term"]
            if isinstance(term,str):
                VAD_dict[term.lower()] = [r[1]['valence'], r[1]['arousal'], r[1]['dominance']]
            return VAD_dict

    def get_prediction_with_diagnosis(self,utterance_1,utterance_2,user_emo,personality,init_mood,model,tokenizer,VAD_dict):
        model.eval()
        with torch.no_grad():
            uttr_input = f"speaker1: {utterance_1} speaker2: {utterance_2}"
            input_ids = tokenizer.encode(uttr_input, add_special_tokens=True, max_length=self.max_len, padding="max_length", truncation=True)
            input_ids_2 = tokenizer.encode(utterance_2, add_special_tokens=True, max_length=self.max_len, padding="max_length", truncation=True)

            attention_masks = [[float(i > 0) for i in input_ids]]

            user_emo_values = [self.Emotion_dict.get(emo, [0.0,0.0,0.0]) for emo in user_emo]
            user_emo_tensor = torch.tensor(user_emo_values,dtype=torch.float32).to(self.device)

            personalities = torch.tensor([personality],dtype=torch.float32).to(self.device)
            personalities = self.personality_to_vad(personalities)

            uttr_vad = self.get_sent_vad_attention(VAD_dict,input_ids_2,tokenizer,user_emo_tensor[0].cpu().numpy())
            input_ids = torch.tensor([input_ids]).to(self.device)
            attention_masks = torch.tensor(attention_masks).to(self.device)
            uttr_vad = torch.tensor([uttr_vad],dtype=torch.float32).to(self.device)
            #init_mood = torch.tensor([init_mood],dtype=torch.float32).to(self.device)
            if isinstance(init_mood, torch.Tensor):
                # すでにTensorなら型・デバイスだけ合わせる
                init_mood = init_mood.to(self.device, dtype=torch.float32)
            else:
                # list や np.ndarray なら新たに Tensor 化
                init_mood = torch.tensor(init_mood, dtype=torch.float32, device=self.device)

            mood_vad,mood_logit,emo_logit = model(input_ids,attention_masks,uttr_vad,user_emo_tensor,personalities,init_mood)
            emo_prob = F.softmax(emo_logit,dim=1)
            pred_flat = np.argmax(emo_prob.detach().cpu().numpy(),axis=1).flatten()
            return pred_flat, mood_vad, mood_logit, emo_prob

    def pred_emo(self,utt1,utt2,emo,pers,mood):
        model = T5EmoMood().to(self.device)
        checkpoint = torch.load("best_model-0.3010.pt",map_location=self.device)
        model.load_state_dict(checkpoint["model_state"],strict=False)

        pred_flat, mood_vad, mood_logit, emo_prob = self.get_prediction_with_diagnosis(utt1,utt2,emo,pers,mood,model,self.tokenizer,self.VAD_dict)

        emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        predicted_emotion = emotion_labels[pred_flat[0]]
        return predicted_emotion,mood_vad

