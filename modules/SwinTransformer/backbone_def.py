
import sys
import yaml
sys.path.append('../../')

from modules.SwinTransformer.Swin_Transformer import SwinTransformer

class BackboneFactory:
    """Factory to produce backbone according the backbone_conf.yaml.
    
    Attributes:
        backbone_type(str): which backbone will produce.
        backbone_param(dict):  parsed params and it's value. 
    """
    def __init__(self, backbone_type, backbone_conf_file):
        self.backbone_type = backbone_type
        with open(backbone_conf_file) as f:
            backbone_conf = yaml.load(f, Loader=yaml.FullLoader)
            self.backbone_param = backbone_conf[backbone_type]
        # print('backbone param:')
        # print(self.backbone_param)

    def get_backbone(self):
        
        if self.backbone_type == 'SwinTransformer':
            img_size = self.backbone_param['img_size']
            patch_size= self.backbone_param['patch_size']
            in_chans = self.backbone_param['in_chans']
            embed_dim = self.backbone_param['embed_dim']
            depths = self.backbone_param['depths']
            num_heads = self.backbone_param['num_heads']
            window_size = self.backbone_param['window_size']
            mlp_ratio = self.backbone_param['mlp_ratio']
            drop_rate = self.backbone_param['drop_rate']
            drop_path_rate = self.backbone_param['drop_path_rate']
            backbone = SwinTransformer(img_size=img_size,
                                        patch_size=patch_size,
                                        in_chans=in_chans,
                                        embed_dim=embed_dim,
                                        depths=depths,
                                        num_heads=num_heads,
                                        window_size=window_size,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=True,
                                        qk_scale=None,
                                        drop_rate=drop_rate,
                                        drop_path_rate=drop_path_rate,
                                        ape=False,
                                        patch_norm=True,
                                        use_checkpoint=False)
        else:
            pass
        return backbone
