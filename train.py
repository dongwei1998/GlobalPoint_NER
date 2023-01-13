# -*- coding: utf-8 -*-
"""
@Time: 2021/8/27 16:12
@Auth: Xhw
@File: entity_extract.py
@Description: 实体抽取.
"""
import os

from utils.data_loader import EntDataset, load_data
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader
import torch
from utils.GlobalPointer import GlobalPointer, MetricsCalculator
from utils import parameter,bio_to_dict
from tqdm import tqdm

args = parameter.parser_opt('train')
device = torch.device(args.device_name)
args.logger.info(args)

if not os.path.exists(args.train_data_path) \
       or not os.path.exists(args.dev_data_path) \
       or not os.path.exists(args.test_data_path) \
       or not os.path.exists(args.ent2id_data_path):
    #bio_to_dict.dict_to_bio(args)
    bio_to_dict.bio_to_dict_conver(args)

#tokenizer
tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_path , do_lower_case=True)

#train_data and val_data
t_data,t_id2ent,t_ent2id = load_data(args.train_data_path,args.ent2id_data_path)
ner_train = EntDataset(data=t_data,ent2id=t_ent2id, tokenizer=tokenizer)
# ner_loader_train = DataLoader(ner_train , batch_size=BATCH_SIZE, collate_fn=ner_train.collate, shuffle=True, num_workers=16)
ner_loader_train = DataLoader(ner_train , batch_size=args.batch_size, collate_fn=ner_train.collate, shuffle=True, num_workers=args.num_workers)
e_data,e_id2ent,e_ent2id = load_data(args.dev_data_path,args.ent2id_data_path)
ner_evl = EntDataset(data=e_data,ent2id=e_ent2id, tokenizer=tokenizer)
# ner_loader_evl = DataLoader(ner_evl , batch_size=BATCH_SIZE, collate_fn=ner_evl.collate, shuffle=False, num_workers=16)
ner_loader_evl = DataLoader(ner_evl , batch_size=args.batch_size, collate_fn=ner_evl.collate, shuffle=False, num_workers=args.num_workers)

#GP MODEL
encoder = BertModel.from_pretrained(args.bert_model_path)
model = GlobalPointer(encoder, len(t_id2ent), 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred alphamind of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12 # mask the pred alphamind of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

def loss_fun(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss


metrics = MetricsCalculator()
max_f, max_recall = 0.0, 0.0
for eo in range(args.num_epochs):
    total_loss, total_f1 = 0., 0.
    for idx, batch in enumerate(ner_loader_train):
        raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
        input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(device), segment_ids.to(device), labels.to(device)
        logits = model(input_ids, attention_mask, segment_ids)
        loss = loss_fun(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
        optimizer.step()
        sample_f1 = metrics.get_sample_f1(logits, labels)
        total_loss+=loss.item()
        total_f1 += sample_f1.item()

        avg_loss = total_loss / (idx + 1)
        avg_f1 = total_f1 / (idx + 1)
        # print("trian_loss:", avg_loss, "\t train_f1:", avg_f1)
        args.logger.info(f"trian_loss: {avg_loss}     train_f1: {avg_f1}")

    with torch.no_grad():
        total_f1_, total_precision_, total_recall_ = 0., 0., 0.
        model.eval()
        for batch in tqdm(ner_loader_evl, desc="Valing"):
            raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
            input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
                device), segment_ids.to(device), labels.to(device)
            logits = model(input_ids, attention_mask, segment_ids)
            try:
                f1, p, r = metrics.get_evaluate_fpr(logits, labels)
            except:
                break
            total_f1_ += f1
            total_precision_ += p
            total_recall_ += r
        avg_f1 = total_f1_ / (len(ner_loader_evl))
        avg_precision = total_precision_ / (len(ner_loader_evl))
        avg_recall = total_recall_ / (len(ner_loader_evl))
        # print("EPOCH：{}\tEVAL_F1:{}\tPrecision:{}\tRecall:{}\t".format(eo, avg_f1,avg_precision,avg_recall))
        args.logger.info("EPOCH：{}\tEVAL_F1:{}\tPrecision:{}\tRecall:{}\t".format(eo, avg_f1,avg_precision,avg_recall))

        if avg_f1 > max_f:
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            torch.save(model.state_dict(), args.save_model.format(eo))
            max_f = avg_f1
        model.train()
