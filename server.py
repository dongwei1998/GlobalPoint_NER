# coding=utf-8
# =============================================
# @Time      : 2023-01-04 14:56
# @Author    : DongWei1998
# @FileName  : server.py.py
# @Software  : PyCharm
# =============================================
from flask import Flask, jsonify, request
from utils import parameter
from transformers import BertModel, BertTokenizerFast
from utils.GlobalPointer import GlobalPointer
import json
import torch
import numpy as np


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
    scores = ner_model(input_ids, attention_mask, token_type_ids)[0].data.cpu().numpy()
    scores[:, [0, -1]] -= np.inf
    scores[:, :, [0, -1]] -= np.inf
    for l, start, end in zip(*np.where(scores > 0)):
        entities.append({"start_idx":new_span[start][0], "end_idx":new_span[end][-1], "type":id2ent[l],"entity":text[new_span[start][0]:new_span[end][-1]]})
    return {"text":text, "entities":entities}
if __name__ == '__main__':
    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False
    args = parameter.parser_opt('server')
    device = torch.device(args.device_name)
    with open(args.ent2id_data_path, 'r', encoding='utf-8') as r1:
        ent2id = json.load(r1)
        id2ent = {}
        for k, v in ent2id.items(): id2ent[v] = k
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_path)
    encoder = BertModel.from_pretrained(args.bert_model_path)
    model = GlobalPointer(encoder, len(id2ent), 64).to(device)
    model.load_state_dict(torch.load(args.save_model, map_location='cuda:0'))
    model.eval()
    @app.route('/app/v1/ner', methods=['POST'])
    def predict():
        try:
            # 参数获取
            infos = request.get_json()
            data_dict = {
                "text": "",
            }
            for k, v in infos.items():
                if v is not None:
                    data_dict[k] = v
            result = NER_RELATION(data_dict['text'],tokenizer= tokenizer, ner_model=model)
            return jsonify({
                        "status": 1,
                        "message": "调用成功",
                        "result": result
                    })
        except Exception as e:
            return jsonify({
                'status': 0,
                'message': e
            })
    # 启动
    app.run(host='0.0.0.0',port=8888)
