#################################
# parse_config.py
# Author: Juha-Matti Rouvinen
# Date: 2023-09-22
# Updated: 2024-01-12
# Version V3
##################################
import yaml


def parse_model_config_and_hyperparams(path,hyp):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    if hyp != None:
        module_defs[0].update(hyp)
    return module_defs

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def parse_hyp_config(path):
    """Parses the hyperparamaters configuration file"""
    options = dict()
    if path.endswith('.yaml'):
        with open(path, 'r') as file:
            data_config = yaml.safe_load(file)
            for x in data_config:
                options[x] = data_config[x]
            '''
            options['optimizer'] = data_config['optimizer']
            options['nesterov'] = data_config['nesterov']
            options['lr_scheduler'] = data_config['lr_scheduler']
            options['lr0'] = data_config['lr0']
            options['lrf'] = data_config['lrf']
            options['dec_gamma'] = data_config['dec_gamma']
            options['momentum'] = data_config['momentum']
            options['weight_decay'] = data_config['weight_decay']
            options['warmup_epochs'] = data_config['warmup_epochs']
            options['warmup_momentum'] = data_config['warmup_momentum']
            options['warmup_bias_lr'] = data_config['warmup_bias_lr']
            options['box'] = data_config['box']
            options['cls'] = data_config['cls']
            options['cls_pw'] = data_config['cls_pw']
            options['obj'] = data_config['obj']
            options['obj_pw'] = data_config['obj_pw']
            options['iou_t'] = data_config['iou_t']
            options['anchor_t'] = data_config['anchor_t']
            options['fl_gamma'] = data_config['fl_gamma']
            options['hsv_h'] = data_config['hsv_h']
            options['hsv_s'] = data_config['hsv_s']
            options['hsv_v'] = data_config['hsv_v']
            options['degrees'] = data_config['degrees']
            options['translate'] = data_config['translate']
            options['scale'] = data_config['scale']
            options['shear'] = data_config['shear']
            options['perspective'] = data_config['perspective']
            options['flipud'] = data_config['flipud']
            options['flipud'] = data_config['flipud']
            '''

    else:
        with open(path, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.strip()
            if line == '' or line.startswith('#') or line.startswith('['):
                continue
            key, value = line.split('=')
            options[key.strip()] = value.strip()
    return options

def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '4'
    if path.endswith('.yaml'):
        with open(path, 'r') as file:
            data_config = yaml.safe_load(file)
        options['train'] = data_config['train']
        options['valid'] = data_config['valid']
        options['eval'] = data_config['eval']
        options['nc'] = data_config['nc']
        options['names'] = data_config['names']
        options['model'] = data_config['model']

    else:
        with open(path, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            key, value = line.split('=')
            options[key.strip()] = value.strip()
    return options

def parse_model_weight_config(path):
    eval_wtrain = ''
    eval_w = ''
    wtrain_float_list = []
    eval_w_float_list = []
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#') or line.startswith('['):
            continue
        key, value = line.split('=')
        if key == "w_train":
            eval_wtrain = value
            eval_wtrain = eval_wtrain.replace('[','')
            eval_wtrain = eval_wtrain.replace(']','')
            eval_wtrain = eval_wtrain.split(',')
            wtrain_float_list = list(map(float, eval_wtrain))
        elif key == "w":
            eval_w = value
            eval_w = eval_w.replace('[','')
            eval_w = eval_w.replace(']','')
            eval_w = eval_w.split(',')
            eval_w_float_list = list(map(float, eval_w))

    return wtrain_float_list,eval_w_float_list

def parse_autodetect_config(path):
    import configparser
    config = configparser.ConfigParser()
    config.read(path)
    autodetect = {}
    autodetect['directory'] = config.get('autodetect', 'directory')
    autodetect['json_path'] = config.get('autodetect', 'json_path')
    autodetect['interval'] = config.get('autodetect', 'interval')
    autodetect['gpu'] = config.getint('autodetect', 'gpu')
    autodetect['n_cpu'] = config.getint('autodetect', 'n_cpu')
    autodetect['batch_size'] = config.getint('autodetect','batch_size')
    autodetect['classes'] = config.get('autodetect', 'classes')
    autodetect['conf_thres'] = config.getfloat('autodetect', 'conf_thres')
    autodetect['nms_thres'] = config.getfloat('autodetect', 'nms_thres')
    autodetect['img_size'] = config.getint('autodetect', 'img_size')
    autodetect['model'] = config.get('autodetect', 'model')
    autodetect['weights'] = config.get('autodetect', 'weights')
    autodetect['hyperparams'] = config.get('autodetect', 'hyperparams')
    autodetect['host'] = config.get('autodetect', 'host')
    autodetect['port'] = config.getint('autodetect', 'port')
    autodetect['username'] = config.get('autodetect', 'username')
    autodetect['password'] = config.get('autodetect', 'password')

    return autodetect


