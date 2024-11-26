import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
from .dvae import Group
from .dvae import DiscreteVAE, Encoder

from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from tools import builder
import copy
import os
import math

class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.down_size = 256
        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        adapter_scalar = "learnable_scalar"
        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config = None):
        super().__init__()

        self.config = config
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x, adapt = None):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if adapt is not None:
            adapt_x = adapt(x, add_residual=False)
        else:
            adapt_x = None

        residual = x
        x = self.drop_path(self.mlp(self.norm2(x)))

        if adapt_x is not None:
            if self.config.ffn_adapt:
                if self.config.ffn_option == 'sequential':
                    x = adapt(x)
                elif self.config.ffn_option == 'parallel':
                    x = x + adapt_x
                else:
                    raise ValueError(self.config.ffn_adapt)

        x = residual + x
        return x

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., tuning_config = None):
        super().__init__()

        self.config = tuning_config
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                config=self.config
                )
            for i in range(depth)])
        # layer norm
        # self.norm = nn.LayerNorm(embed_dim)
        self._device = self.config._device
        self.adapter_list = []
        self.cur_adapter = nn.ModuleList()
        self.get_new_adapter()

    def forward(self, x, pos):
        for idx, block in enumerate(self.blocks):
            x = block(x + pos, self.cur_adapter[idx])
            # x = block(x + pos, None)
        return x #self.norm(x)

    def forward_test(self, x_embed, pos, use_init_ptm=False):

        features = []

        if use_init_ptm:
            x = copy.deepcopy(x_embed)
            for idx, block in enumerate(self.blocks):
                x = block(x + pos, self.cur_adapter[idx])
            #x = self.norm(x)
            features.append(x)

        for i in range(len(self.adapter_list)):
            x = copy.deepcopy(x_embed)
            for idx, block in enumerate(self.blocks):
                adapt = self.adapter_list[i][idx]
                x = block(x + pos, adapt)
            #x = self.norm(x)
            features.append(x)

        x = copy.deepcopy(x_embed)
        for idx, block in enumerate(self.blocks):
            adapt = self.cur_adapter[idx]
            x = block(x + pos, adapt)
        #x = self.norm(x)
        features.append(x)

        return features

    def forward_proto(self, x_embed, pos, adapt_index):

        if adapt_index == -1:
            x = copy.deepcopy(x_embed)
            for idx, block in enumerate(self.blocks):
                x = block(x + pos, self.cur_adapter[idx])
            return x #self.norm(x)

        i = adapt_index
        x = copy.deepcopy(x_embed)
        for idx, block in enumerate(self.blocks):
            if i < len(self.adapter_list):
                adapt = self.adapter_list[i][idx]
            else:
                adapt = self.cur_adapter[idx]
            x = block(x + pos, adapt)
        return x #self.norm(x)

    def get_new_adapter(self):
        config = self.config
        self.cur_adapter = nn.ModuleList()
        if config.ffn_adapt:
            for i in range(len(self.blocks)):
                adapter = Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num,
                                        init_option=config.ffn_adapter_init_option,
                                        adapter_scalar=config.ffn_adapter_scalar,
                                        adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                        ).to(self._device)
                self.cur_adapter.append(adapter)
            self.cur_adapter.requires_grad_(True)
        else:
            print("====Not use adapter===")

    def add_adapter_to_list(self):
        self.adapter_list.append(copy.deepcopy(self.cur_adapter.requires_grad_(False)))
        self.get_new_adapter()

@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth 
        self.drop_path_rate = config.drop_path_rate 
        self.cls_dim = config.cls_dim 
        self.num_heads = config.num_heads 

        self.group_size = config.group_size
        self.num_group = config.num_group
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims =  config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()
        
    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
    
    def get_loss_acc(self, pred, gt, smoothing=True):
        # import pdb; pdb.set_trace()
        gt = gt.contiguous().view(-1).long()

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = self.loss_ce(pred, gt.long())

        pred = pred.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))

        return loss, acc * 100


    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]


        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger = 'Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger = 'Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger = 'Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger = 'Transformer'
            )

        print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger = 'Transformer')


    def forward(self, pts):
        # divide the point clo  ud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:,0], x[:, 1:].max(1)[0]], dim = -1)
        ret = self.cls_head_finetune(concat_f)
        return ret

class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate 
        self.cls_dim = config.transformer_config.cls_dim 
        self.replace_pob = config.transformer_config.replace_pob
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[Transformer args] {config.transformer_config}', logger = 'dVAE BERT')
        # define the encoder
        self.encoder_dims =  config.dvae_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)
        try:
            self.mask_rand = config.mask_rand
        except:
            self.mask_rand = False
        
        # define the learnable tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        # pos embedding for each patch 
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        # define the transformer blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
            tuning_config = config
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        '''
        # head for token classification
        self.num_tokens = config.dvae_config.num_tokens
        self.lm_head = nn.Linear(self.trans_dim, self.num_tokens)
        # head for cls contrast
        self.cls_head = nn.Sequential(
            nn.Linear(self.trans_dim, self.cls_dim),
            nn.GELU(),
            nn.Linear(self.cls_dim, self.cls_dim)
        )  
        # initialize the learnable tokens
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)
        '''

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _prepare_encoder(self, dvae_ckpt):
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        encoder_ckpt = {k.replace("encoder.", ""): v for k, v in base_ckpt.items() if 'encoder' in k}

        self.encoder.load_state_dict(encoder_ckpt, strict=True)
        print_log(f'[Encoder] Successful Loading the ckpt for encoder from {dvae_ckpt}', logger = 'dVAE BERT')

    def forward(self, neighborhood, center, return_all_tokens = False, only_cls_tokens = False, adapt_index = None, test = False, use_init_ptm = False):
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens) # dim 256
        batch_size, seq_len, _ = group_input_tokens.size()
        # prepare cls and mask
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  
        cls_pos = self.cls_pos.expand(batch_size, -1, -1)

        # add pos embedding
        pos = self.pos_embed(center) # dim 384
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1) #  dim 384
        pos = torch.cat((cls_pos, pos), dim=1) # dim 384
        # transformer
        if adapt_index is None:
            if not test:
                x = self.blocks(x, pos)
                # x = self.norm(x)
            else:
                x = self.blocks.forward_test(x, pos, use_init_ptm)
        else:
            x = self.blocks.forward_proto(x, pos, adapt_index)
        # only return the cls feature, for moco contrast
        if only_cls_tokens:
            if not test:
                return x[:, 0]
            else:
                return x
        logits = self.lm_head(x[:, 1:])
        if return_all_tokens:
            return self.cls_head(x[:, 0]), logits

    def add_adapter_to_list(self):
        self.blocks.add_adapter_to_list()


@MODELS.register_module()
class Point_BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_BERT] build dVAE_BERT ...', logger ='Point_BERT')
        self.config = config
        self.return_all_tokens = config.transformer_config.return_all_tokens
        self.transformer_q = MaskTransformer(config)
        self.transformer_q._prepare_encoder(self.config.dvae_config.ckpt)

        self.group_size = config.dvae_config.group_size
        self.num_group = config.dvae_config.num_group

        print_log(f'[Point_BERT Group] cutmix_BERT divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_BERT')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

    def forward(self, pts, test=False, use_init_ptm=False):
        neighborhood, center = self.group_divider(pts)
        if not test:
            output = self.transformer_q(neighborhood, center, only_cls_tokens = True, test = False)
        else:
            q_cls_feature = self.transformer_q(neighborhood, center, only_cls_tokens=True, test = True ,use_init_ptm = use_init_ptm)
            output = torch.Tensor().to(q_cls_feature[0].device)
            for x in q_cls_feature:
                cls = x[:, 0, :]
                output = torch.cat((
                    output,
                    cls
                ), dim=1)

        return output

    def forward_proto(self, pts, adapt_index):
        neighborhood, center = self.group_divider(pts)
        q_cls_feature = self.transformer_q(neighborhood, center, only_cls_tokens = True, adapt_index = adapt_index)
        return q_cls_feature

    def add_adapter_to_list(self):
        self.transformer_q.add_adapter_to_list()

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    #tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    #torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    #output = torch.cat(tensors_gather, dim=0)
    #return output
    return tensor

def point_bert_ease(modelconfig, args, pretrained=False):
    model = builder.model_builder(modelconfig)
    if args['use_gpu']:
        model.to(args['local_rank'])
    if not os.path.exists(modelconfig.ckpt):
        print(f'[RESUME INFO] no checkpoint file from path {modelconfig.ckpt}...')
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args['local_rank']}
    state_dict = torch.load(modelconfig.ckpt, map_location=map_location)
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    # model.load_state_dict(base_ckpt, strict=False)
    # model = nn.DataParallel(model).cuda()
    incompatible = print_model_keys_and_loading_status(model, base_ckpt)
    print('Load PointBert Model Finished')
    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in incompatible.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model


def print_model_keys_and_loading_status(model, state_dict):
    # 获取模型的所有参数键
    model_keys = set(model.state_dict().keys())

    # 加载状态字典并获取返回的_IncompatibleKeys对象
    incompatible = model.load_state_dict(state_dict, strict=False)

    # 获取未加载和意外加载的键
    missing_keys = set(incompatible.missing_keys)
    unexpected_keys = set(incompatible.unexpected_keys)

    # 遍历模型的所有参数键
    for key in model_keys:
        # 检查键是否被加载
        is_loaded = key not in missing_keys
        print(f"{key}: {is_loaded}")

    return incompatible
    '''
    # 遍历意外加载的键
    for key in unexpected_keys:
        print(f"{key}: True (Unexpected)")
    '''