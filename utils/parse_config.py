#################################
# parse_config.py
# Author: Juha-Matti Rouvinen
# Date: 2023-09-22
# Updated: 2023-11-02
#
##################################

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
    options['num_workers'] = '10'
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


