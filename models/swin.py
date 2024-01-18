import timm
from timm.models.swin_transformer import _cfg


def swin(num_classes=5, checkpoint_path=None):
    model = timm.create_model(model_name="swin_base_patch4_window7_224.ms_in22k_ft_in1k", num_classes=num_classes,
                              img_size=224, pretrained=False, pretrained_cfg=_cfg(url='', file=checkpoint_path))
    return model

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == "__main__":
    model = swin()
    print(get_parameter_number(model))
    # print(model)
