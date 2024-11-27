import copy
from torch import nn
from backbone_3DShape.linears import CosineLinear

def get_backbone(args, pretrained=False, modelconfig = None):

    name = args["backbone_type"].lower()
    if '_ease' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "ease" :
            from backbone_3DShape import Point_BERT
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option= "parallel", # "sequential",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model= 768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
                _device = args["device"][0]
            )
            if name == "pointbert_ease":
                modelconfig.update(tuning_config)
                model = Point_BERT.point_bert_ease(modelconfig, args)
                model.out_dim= 768 #384
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    else:
        raise NotImplementedError("Unknown type {}".format(name))

# def get_backbone_pointbert(args, pretrained=False):


class BaseNet(nn.Module):
    def __init__(self, args, pretrained, modelconfig):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained, modelconfig = modelconfig)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out, x

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

class EaseNet(BaseNet):
    def __init__(self, args, pretrained=True, modelconfig = None):
        super().__init__(args, pretrained, modelconfig=modelconfig)
        self.args = args
        self.inc = args["increment"]
        self.last_inc = args["increment"]
        self.init_cls = args["init_cls"]
        self.nb_tasks = -1
        self._cur_task = -1
        if modelconfig is not None:
            self.out_dim = modelconfig.transformer_config.trans_dim
        else:
            self.out_dim =  self.backbone.out_dim
        self.fc = None
        self.use_init_ptm = args["use_init_ptm"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]
            
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            # print(name)
    
    @property
    def feature_dim(self):
        if self.use_init_ptm:
            return self.out_dim * (self._cur_task + 2)
        else:
            return self.out_dim * (self._cur_task + 1)

    # (proxy_fc = cls * dim)
    def update_fc(self, nb_classes, inc = 0, use_exemplars = False):
        self._cur_task += 1
        if inc == 0:
            inc = self.inc
        if self._cur_task == 0:
            self.proxy_fc = self.generate_fc(self.out_dim, self.init_cls).to(self._device)
        else:
            if use_exemplars == False:
                self.proxy_fc = self.generate_fc(self.out_dim, inc).to(self._device)
            else:
                self.proxy_fc = self.generate_fc(self.out_dim, nb_classes).to(self._device)
        
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        fc.reset_parameters_to_zero()
        
        if self.fc is not None:
            old_nb_classes = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            fc.weight.data[ : old_nb_classes, : -self.out_dim] = nn.Parameter(weight)
        del self.fc
        self.fc = fc
    
    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc
    
    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x, test=False):
        if test == False:
            x = self.backbone.forward(x, False)
            out = self.proxy_fc(x)
        else:
            x = self.backbone.forward(x, True, use_init_ptm=self.use_init_ptm)
            if self.args["moni_adam"] or (not self.args["use_reweight"]):
                out = self.fc(x)
            else:
                if self._cur_task == self.nb_tasks - 1:
                    out = self.fc.forward_reweight(x, cur_task=self._cur_task, alpha=self.alpha, init_cls=self.init_cls, last_cls=self.last_cls, inc=self.inc, use_init_ptm=self.use_init_ptm, beta=self.beta)
                else:
                    out = self.fc.forward_reweight(x, cur_task=self._cur_task, alpha=self.alpha, init_cls=self.init_cls, inc=self.inc, use_init_ptm=self.use_init_ptm, beta=self.beta)
            # out = self.proxy_fc(x)
        out.update({"features": x})
        return out

    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())