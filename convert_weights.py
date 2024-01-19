import torch
from models import Darknet
from utils.parse_config import parse_hyp_config
print(f'---- Loading hyperparameters ----')
hyp_config = parse_hyp_config("C:/Users/Juha/PycharmProjects/YoloV3_PyTorch/config/hyp.cfg")
print(f'---- Loaded ----')
print(f'---- Loading model config ----')
model = Darknet("C:/Users/Juha/Documents/AI/Models/Orion/Orion-tiny_v4/Orion-tiny_v2.cfg",hyp_config)
print(f'---- Loaded ----')
#model.load_state_dict(torch.load("C:/Users/Juha/PycharmProjects/YoloV3_PyTorch/weights/Nova_2023_09_25_08_50_04_ckpt_best.pth", map_location=torch.device('cpu'))) # for loading model on cpu
print(f'---- Reading weights from model ----')
model.load_state_dict(torch.load("C:/Users/Juha/Documents/AI/Models/Orion/experiments/exp12/checkpoints/Orion-tiny_832_v4_2023_12_01_12_40_50_ckpt_best.pth", map_location=torch.device('cpu'))) # for loading model on cpu
print(f'---- Converting weights to *.weights ----')
model.save_darknet_weights('C:/Users/Juha/PycharmProjects/YoloV3_PyTorch/weights/Orion-tiny_832_v4_exp12.weights', cutoff=-1)
print(f"---- successfully converted .pth to .weights ----")