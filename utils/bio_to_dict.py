# coding=utf-8
# =============================================
# @Time      : 2023-01-05 14:15
# @Author    : DongWei1998
# @FileName  : bio_to_dict.py
# @Software  : PyCharm
# =============================================
import os,json
from utils import parameter

def dict_to_bio(args):
    if not os.path.exists(f'./datasets/{args.model_class}'):
        os.mkdir(f'./datasets/{args.model_class}')
    ent2id_data_path = f'./datas/{args.model_class}/ent2id_data.json'
    train_data_path = f'./datas/{args.model_class}/train_data.json'
    dev_data_path = f'./datas/{args.model_class}/dev_data.json'
    test_data_path = f'./datas/{args.model_class}/test_data.json'
    with open(ent2id_data_path,'r',encoding='utf-8') as r1:
        ent2id = json.load(r1)
        id2ent = {}
        for k, v in ent2id.items(): id2ent[v] = k
    split_num= 0
    text_train = open(f'./datasets/{args.model_class}/text_train.txt', 'w', encoding='utf-8')
    labels_train = open(f'./datasets/{args.model_class}/labels_train.txt', 'w', encoding='utf-8')
    text_val = open(f'./datasets/{args.model_class}/text_val.txt', 'w', encoding='utf-8')
    labels_val = open(f'./datasets/{args.model_class}/labels_val.txt', 'w', encoding='utf-8')
    text_test = open(f'./datasets/{args.model_class}/text_test.txt', 'w', encoding='utf-8')
    labels_test = open(f'./datasets/{args.model_class}/labels_test.txt', 'w', encoding='utf-8')
    for file_path in [train_data_path]:
        with open(file_path,'r',encoding='utf-8') as r:
            text_list = r.readlines()
            for text_str in text_list:
                split_num += 1
                d = json.loads(text_str)
                sen_list = ['' for i in range(len(d['text']))]
                label_list = ['' for i in range(len(d['text']))]
                for e in d['label']:
                    label = e
                    for k,v in d['label'][e].items():
                        for s_e in v:
                            start = s_e[0]
                            end = s_e[1]
                            for idx,z in enumerate(d['text']):
                                sen_list[idx] = z
                                if idx< start or idx > end:
                                    if label_list[idx] == '':
                                        label_list[idx] = 'O'
                                else:
                                    if idx == start:
                                        label_list[idx] = F'B-{label}'
                                    else:
                                        label_list[idx] = F'I-{label}'
                # 写入文件
                sen_str = ' '.join(sen_list)+'\n'
                label_str = ' '.join(label_list)+'\n'
                if split_num <= len(text_list) * 0.7:
                    text_train.write(sen_str)
                    labels_train.write(label_str)
                elif len(text_list) * 0.7 < split_num <= len(text_list) * 0.9:
                    text_val.write(sen_str)
                    labels_val.write(label_str)
                else:
                    text_test.write(sen_str)
                    labels_test.write(label_str)
    text_train.close()
    labels_train.close()
    text_val.close()
    labels_val.close()
    text_test.close()
    labels_test.close()

def witer_json(r):
    fin_info = []
    info_dict = {
        "text": "",
        "entities": []
    }
    text_list = r.readlines()
    for text_str in text_list:
        d = json.loads(text_str)
        info_dict['text'] = d['text']
        for e in d['label']:
            for k, v in d['label'][e].items():
                for s_e in v:
                    start = s_e[0]
                    end = s_e[1]
                    info_dict['entities'].append({
                        "start_idx": start,
                        "end_idx": end,
                        "type": e,
                        "entity": d['text'][start:end]
                    })
        fin_info.append(info_dict)
        info_dict = {
            "text": "",
            "entities": []
        }
    return fin_info


def bio_to_dict_conver(args):
    train_data = open(f'{args.train_data_dir}/train_data.json', 'w', encoding='utf-8')
    dev_data = open(f'{args.train_data_dir}/dev_data.json', 'w', encoding='utf-8')
    test_data = open(f'{args.train_data_dir}/test_data.json', 'w', encoding='utf-8')
    ent2id_data = open(f'{args.train_data_dir}/ent2id_data.json', 'w', encoding='utf-8')

    text_train = open(f'{args.train_data_dir}/text_train.txt', 'r', encoding='utf-8')
    labels_train = open(f'{args.train_data_dir}/labels_train.txt', 'r', encoding='utf-8')
    text_val = open(f'{args.train_data_dir}/text_val.txt', 'r', encoding='utf-8')
    labels_val = open(f'{args.train_data_dir}/labels_val.txt', 'r', encoding='utf-8')
    text_test = open(f'{args.train_data_dir}/text_test.txt', 'r', encoding='utf-8')
    labels_test = open(f'{args.train_data_dir}/labels_test.txt', 'r', encoding='utf-8')





    info_dict = {
        "text": "",
        "entities": []
    }
    fin_info = []

    ent2id_dict = {}
    ent2id_idx = 0
    for k,v in dict(zip([text_train,text_val,text_test],[labels_train,labels_val,labels_test])).items():
        label_lists = [l for l in v.readlines()]
        for idx,sen in enumerate(k.readlines()):
            text_str = sen.replace(' ','')
            info_dict['text'] = text_str
            try:
                label_list = label_lists[idx].split(' ')
            except:
                continue
            start,end,e = '','',''
            for lidx,l in enumerate(label_list):
                if l != 'O' and 'B' in l:
                    start = lidx
                    e = l.replace('B-','')
                    if e not in ent2id_dict.keys():
                        ent2id_dict[e] = ent2id_idx
                        ent2id_idx +=1
                elif l == 'O' and 'I' in label_list[lidx-1] and lidx != 0:
                    end = lidx-1
                else:
                    continue
                if start != '' and end != '' and e != '':
                    info_dict['entities'].append({
                        "start_idx": start,
                        "end_idx": end,
                        "type": e,
                        "entity": text_str[start:end]
                    })
                    start, end, e = '', '', ''
            fin_info.append(info_dict)
            info_dict = {
                "text": "",
                "entities": []
            }

    # print(fin_info)
    # 写入文件
    t_list= []
    d_list = []
    x_list = []
    split_num = 0
    for idx,dic_s in enumerate(fin_info):
        split_num+=1
        if split_num <= len(fin_info) * 0.7:
            t_list.append(dic_s)
        elif len(fin_info) * 0.7 < split_num <= len(fin_info) * 0.9:
            d_list.append(dic_s)
        else:
            x_list.append(dic_s)
    json.dump(t_list, train_data, indent=4, ensure_ascii=False)
    json.dump(d_list, dev_data, indent=4, ensure_ascii=False)
    json.dump(x_list, test_data, indent=4, ensure_ascii=False)
    json.dump(ent2id_dict, ent2id_data, indent=4, ensure_ascii=False)






    train_data.close()
    dev_data.close()
    test_data.close()
    text_train.close()
    labels_train.close()
    text_val.close()
    labels_val.close()
    text_test.close()
    labels_test.close()




if __name__ == '__main__':
    args = parameter.parser_opt('train')
    # print(args)
    # dict_to_bio(args)
    bio_to_dict_conver(args)
