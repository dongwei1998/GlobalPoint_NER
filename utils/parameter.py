# coding=utf-8
# =============================================
# @Time      : 2023-01-04 15:02
# @Author    : DongWei1998
# @FileName  : parameter.py.py
# @Software  : PyCharm
# =============================================
import os
from easydict import EasyDict
from dotenv import load_dotenv,find_dotenv
import logging.config



# 创建路径
def check_directory(path, create=True):
    flag = os.path.exists(path)
    if not flag:
        if create:
            os.makedirs(path)
            flag = True
    return flag


def data_conver(args):



    return args.train_data_path,args.dev_data_path,args.test_data_path,args.ent2id_data_path


def parser_opt(model):
    load_dotenv(find_dotenv())  # 将.env文件中的变量加载到环境变量中
    args = EasyDict()
    logging.config.fileConfig(os.environ.get("logging_ini"))
    args.logger = logging.getLogger('model_log')
    if model =='train':
        args.model_class = os.environ.get('model_class') # 训练哪一个模型 【电费类合同、其他类合同等】
        args.train_data_dir = os.environ.get('train_data_dir')    # 数据主目录
        args.bert_model_path = os.environ.get('bert_model_path')
        args.ent2id_data_path = os.path.join(args.train_data_dir, 'ent2id_data.json')
        args.train_data_path = os.path.join(args.train_data_dir, 'train_data.json')
        args.dev_data_path = os.path.join(args.train_data_dir, 'dev_data.json')
        args.test_data_path = os.path.join(args.train_data_dir, 'test_data.json')
        args.device_name = os.environ.get('device_name')
        args.batch_size = int(os.environ.get('batch_size'))
        args.num_epochs = int(os.environ.get('num_epochs'))
        args.num_workers = int(os.environ.get('num_workers'))
        args.learning_rate = float(os.environ.get('learning_rate'))
        args.save_model = os.path.join(os.environ.get('output_dir'),os.environ.get('save_model_name'))
    elif model == 'server':
        args.model_class = os.environ.get('model_class') # 训练哪一个模型 【电费类合同、其他类合同等】
        args.train_data_dir = os.environ.get('train_data_dir')    # 数据主目录
        args.bert_model_path = os.environ.get('bert_model_path')
        args.ent2id_data_path = os.path.join(args.train_data_dir,'ent2id_data.json')
        args.device_name = os.environ.get('device_name')
        args.num_workers = int(os.environ.get('num_workers'))
        args.save_model = os.path.join(os.environ.get('output_dir'),os.environ.get('save_model_name'))
    elif model == 'env':
        pass
    else:
        raise print('请给定model参数，可选范围【train | env | server】')
    if not os.path.exists(os.environ.get('output_dir')):
        os.mkdir(os.environ.get('output_dir'))
    return args

if __name__ == '__main__':
    args = parser_opt('train')
