# -*- coding: utf-8 -*-
"""
@Time: 2021/9/1 14:29
@Auth: Xhw
@Description:
"""
from transformers import BertModel, BertTokenizerFast
from utils.GlobalPointer import GlobalPointer
import json
import torch
import numpy as np

bert_model_path = '../RoBERTa_zh_Large_PyTorch'  #your RoBert_large path
save_model_path = 'alphamind/ent_model.pth'
device = torch.device("cuda:0")

max_len = 256
ent2id_path = '../datas/dianfei/ent2id_data.json'
with open(ent2id_path, 'r', encoding='utf-8') as r1:
    ent2id = json.load(r1)
    id2ent = {}
    for k, v in ent2id.items(): id2ent[v] = k

tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)
encoder =BertModel.from_pretrained(bert_model_path)
model = GlobalPointer(encoder, len(id2ent) , 64).to(device)
model.load_state_dict(torch.load(save_model_path, map_location='cuda:0'))
model.eval()

def NER_RELATION(text, tokenizer, ner_model,  max_len=256):
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=max_len)["offset_mapping"]
    new_span, entities= [], []
    for i in token2char_span_mapping:
        if i[0] == i[1]:
            new_span.append([])
        else:
            if i[0] + 1 == i[1]:
                new_span.append([i[0]])
            else:
                new_span.append([i[0], i[-1] - 1])

    encoder_txt = tokenizer.encode_plus(text, max_length=max_len)
    input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).cuda()
    token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).cuda()
    attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).cuda()
    scores = model(input_ids, attention_mask, token_type_ids)[0].data.cpu().numpy()
    scores[:, [0, -1]] -= np.inf
    scores[:, :, [0, -1]] -= np.inf
    for l, start, end in zip(*np.where(scores > 0)):
        entities.append({"start_idx":new_span[start][0], "end_idx":new_span[end][-1], "type":id2ent[l],"entity":text[new_span[start][0]:new_span[end][-1]]})
    return {"text":text, "entities":entities}

if __name__ == '__main__':
    all_ = []
    with open('../datas/dianfei/test_data.json', 'r', encoding='utf-8') as r:
        text_list = r.readlines()
        for idx,text_str in enumerate(text_list):
            d = json.loads(text_str)
            all_.append(NER_RELATION(d['text'],tokenizer= tokenizer, ner_model=model))
            print(idx)
    json.dump(
        all_,
        open('alphamind/dianfei.json', 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )