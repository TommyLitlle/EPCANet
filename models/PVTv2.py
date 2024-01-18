import timm
from timm.models.pvt_v2 import _cfg


def pvt_v2(num_classes=5, checkpoint_path=None):
    model = timm.create_model(model_name="pvt_v2_b0.in1k", num_classes=num_classes,
                              pretrained=False, pretrained_cfg=_cfg(url='', file=checkpoint_path))
    return model
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == "__main__":
    model = pvt_v2()
    print(get_parameter_number(model))
    # print(model)
