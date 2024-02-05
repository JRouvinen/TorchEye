import os

#################################
# folder_management.py
# Author: Juha-Matti Rouvinen
# Date: 2024-02-02
# Updated: 2024-02-02
# Version V1
##################################
def create_exp_folder_structure(logdir, model_name):
    local_path = os.getcwd()
    model_logs_path = os.path.join(local_path, logdir, model_name)
    # Create model named log folder in logs folder

    # Check if logs folder exists
    logs_path_there = os.path.exists(model_logs_path)
    if not logs_path_there:
        os.mkdir(model_logs_path + '/')

    # Check if images folder exists
    imgs_logs = '/images/'
    imgs_path_there = os.path.exists(model_logs_path + imgs_logs)
    if not imgs_path_there:
        os.mkdir(model_logs_path + imgs_logs)
    model_imgs_logs_path = model_logs_path + imgs_logs

    # Check if images epoch data exists
    imgs_epoch_logs = '/images/epoch_data/'
    imgs_epoch_logs_there = os.path.exists(model_logs_path + imgs_epoch_logs)
    if not imgs_epoch_logs_there:
        os.mkdir(model_logs_path + imgs_epoch_logs)
    model_imgs_epoch_logs_path = model_logs_path + imgs_epoch_logs

    # Check if checkpoints folder exists
    model_ckpt = '/checkpoints/'
    model_ckpt_path_there = os.path.exists(model_logs_path + model_ckpt)
    if not model_ckpt_path_there:
        os.mkdir(model_logs_path + model_ckpt)
    model_ckpt_logs_path = model_logs_path + model_ckpt

    return model_logs_path, model_imgs_logs_path, model_ckpt_logs_path, model_imgs_epoch_logs_path
