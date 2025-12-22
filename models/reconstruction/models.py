import math
import torch
import torch.nn as nn
from functools import partial
from .modules import Block, _cfg, RelativePositionBias
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from einops import rearrange


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'vit_base_patch16_224_8k_vocab',
    'vit_large_patch16_224_8k_vocab',
]

class PatchLevelReconstructionModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=10,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02,
                 outplane=272, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = (img_size // patch_size, img_size // patch_size)
        num_patches = self.window_size[0] * self.window_size[1]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.window_size, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.window_size if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(depth)])
        self.init_std = init_std

        self.input_proj = nn.Linear(outplane, embed_dim)
        self.output_proj = nn.Linear(embed_dim, outplane)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=16)
        self.criterion_mse = nn.MSELoss()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward(self, input, tokenimg_fe, bool_masked_pos):
        feature_align = input["feature_align"]
        feature_tokens = rearrange(
            feature_align, "b c h w -> b (h w) c "
        )
        batch_size = feature_tokens.shape[0]
        seq_len = feature_tokens.shape[1]
        feature_tokens = self.input_proj(feature_tokens)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)


        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        feature_tokens = feature_tokens * (1 - w) + mask_token * w

        feature_tokens = torch.cat((cls_tokens, feature_tokens), dim=1)

        if self.pos_embed is not None:
            feature_tokens = feature_tokens + self.pos_embed
        feature_tokens = self.pos_drop(feature_tokens)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        for blk in self.blocks:
            feature_tokens = blk(feature_tokens, rel_pos_bias=rel_pos_bias)

        feature_rec_tokens = self.output_proj(feature_tokens)
        feature_rec_tokens = feature_rec_tokens[:, 1:]
        feature_rec = rearrange(
            feature_rec_tokens, "b (h w) c -> b c h w", h=self.window_size[0]
        )
        pred = torch.sqrt(
            torch.sum((feature_rec - tokenimg_fe["feature_align"]) ** 2, dim=1, keepdim=True)
        )

        patches_distribution = torch.t(pred.reshape(batch_size, 196))

        rmse_patches_score = torch.mean(patches_distribution, dim=1, keepdim=True)
        rmse_patches_loss = torch.mean(rmse_patches_score)
        rmse_patches_score_map = rmse_patches_score.reshape(1, 1, 14, 14)
        rmse_patches_score_map = self.upsample(rmse_patches_score_map)

        mseloss = self.criterion_mse(feature_rec, tokenimg_fe["feature_align"])
        return {
            "feature_rec": feature_rec,
            "feature_align": feature_align,
            "patches_distribution": patches_distribution,
            "rmse_patches_score": rmse_patches_score,
            "rmse_patches_score_map": rmse_patches_score_map,
            "mseLoss": mseloss,
            "rmse_patches_loss": rmse_patches_loss
        }



@register_model
def vit_base_patch16_224_8k_vocab(pretrained=False, **kwargs):
    model = PatchLevelReconstructionModel(
        patch_size=16, embed_dim=768, depth=10, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vit_large_patch16_224_8k_vocab(pretrained=False, **kwargs):
    model = PatchLevelReconstructionModel(
        patch_size=16, embed_dim=256, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
