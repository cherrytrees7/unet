import torch
import torch.nn as nn
from nets.attention_net import AttU_Net
from nets.rac2_model import unetUp_RAC2
from nets.rac3_model import unetUp_RAC3
from nets.rac4_model import unetUp_RAC4
from nets.unet import U_Net
from nets.unet1 import Unet
from nets.unet_plus import NestedUNet, R2AttU_Net

# 实例化模型
models = {
    "Unet": Unet(),
    "U_Net": U_Net(),
    "AttU_Net": AttU_Net(),
    "Unet_plus": NestedUNet(),
    "Unet_RAC2": unetUp_RAC2(),
    "Unet_RAC3": unetUp_RAC3(),
    "Unet_RAC4": unetUp_RAC4(),
    "Unet_R2AttU_Net": R2AttU_Net()
}

# 计算每个模型的可训练参数数量
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 创建一个字典来存储模型和它们的可训练参数数量
models_params = {name: count_trainable_parameters(model) for name, model in models.items()}

# 按照可训练参数数量排序模型
sorted_models = sorted(models_params.items(), key=lambda x: x[1])

# 打印排序后的模型及其参数数量
for model_name, params in sorted_models:
    print(f"{model_name}: {params} trainable parameters")
