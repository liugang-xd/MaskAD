import warnings

warnings.filterwarnings('ignore')
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from datasets.test import MVTEC
from protoflow_module.utils import measure_rocaucs
from protoflow_module.protoflow import ProtoFlow
from utils.mask_generator import RandomMaskGenerator, BlockWiseMaskGenerator
from timm.models import create_model
from models.reconstruction import models
from models.Sinkhorn_distance import SinkhornDistance
import yaml
from easydict import EasyDict
from models.model_helper import ModelHelper
from utils.misc_helper import update_config
import os
import cv2

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def prepare_model(args, ckpt_dir):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
    )
    checkpoint = torch.load(ckpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def get_feature_model():
    with open("./config.yaml") as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    config = update_config(config)
    effmodel = ModelHelper(config.net)
    return effmodel.eval()


def evaluation(args, model, protoflow, effmodel, OT_cos):
    device = torch.device(args.device)
    args.window_size = (args.input_size // 16, args.input_size // 16)

    dataset = MVTEC(args.data_path, class_name=args.class_name, train=False,
                    norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225], img_size=224, crp_size=224)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    refence_path = os.path.join(args.ref_data_path, args.class_name, "ref.pt")
    reference = torch.load(refence_path)

    p_batch, r_batch, b_batch, abnomalities = ge_msking(protoflow, data_loader, args)
    upsample = nn.UpsamplingBilinear2d(scale_factor=16)

    gt_label_list = []
    gt_mask_list = []
    score_list = []
    img_types = []
    img_path_list = []

    for i, (img, token_img, label, mask, img_path, img_type) in \
            enumerate(tqdm(data_loader, 'Evaluation Dataset->{}.{}'.format('MVTEC', args.class_name))):
        gt_label_list.append(label.cpu().numpy())
        gt_mask_list.append(mask.squeeze(0).cpu().numpy())
        img_types.append(img_type)
        img_path_list.append(img_path)

        img = img.to(device, non_blocking=True)
        token_img = token_img.to(device, non_blocking=True)

        ref_img = reference["img_ref"][i].to(device, non_blocking=True)
        ref_token_img = reference["tokenimg_ref"][i].to(device, non_blocking=True)

        batch_token_images = token_img.repeat(12, 1, 1, 1)
        batch_images = img.repeat(12, 1, 1, 1)

        ref_batch_token_images = ref_token_img.repeat(12, 1, 1, 1)
        ref_batch_images = ref_img.repeat(12, 1, 1, 1)

        batch_mask = torch.cat([p_batch[i], r_batch[i], b_batch[i]], dim=0).to(device)
        batch_mask_2 = batch_mask.repeat(2, 1)
        effmodel.eval()
        with torch.no_grad():
            input = {"image": batch_token_images}
            tokenimg_fe = effmodel(input)

            input = {"image": batch_images}
            img_fe = effmodel(input)

            input = {"image": ref_batch_token_images}
            ref_tokenimg_fe = effmodel(input)

            input = {"image": ref_batch_images}
            ref_img_fe = effmodel(input)

            img_fe["feature_align"] = torch.cat((img_fe["feature_align"], ref_img_fe["feature_align"]), dim=0)
            tokenimg_fe["feature_align"] = torch.cat((tokenimg_fe["feature_align"], ref_tokenimg_fe["feature_align"]),
                                                     dim=0)

            outputs = model(img_fe, tokenimg_fe, bool_masked_pos=batch_mask_2)
            patches_distribution = outputs["patches_distribution"].to(device)
            chunked_dis = torch.chunk(patches_distribution, 2, dim=1)
            dis_patches1 = chunked_dis[0]
            dis_patches2 = chunked_dis[1]
            OT_dist, cost, pi, C = OT_cos(dis_patches1, dis_patches2)
            OT_score = (OT_dist * 10 ** 4).unsqueeze(1)
            rmse_patches_score = torch.mean(dis_patches1, dim=1, keepdim=True)
            patches_score = OT_score * args.OT_w + rmse_patches_score * (1.0 - args.OT_w)
            patches_score = patches_score.reshape(1, 1, 14, 14)
            patches_score = upsample(patches_score)
            patches_score = patches_score.squeeze()
            score = patches_score.cpu().numpy()
            score = score * abnomalities[i]
            score = gaussian_filter(score, sigma=7)

        score_list.append(score)
    scores = np.stack(score_list, axis=0)
    labels = np.asarray(gt_label_list)
    masks = np.asarray(gt_mask_list, dtype=bool)
    img_scores = np.max(scores.reshape(scores.shape[0], -1), axis=-1)
    image_auc, pixel_auc, _ = measure_rocaucs(labels, masks, img_scores, scores)
    print(f'Category {args.class_name}: Image-AUC: {image_auc}, Pixel AUC: {pixel_auc}')
    return image_auc, pixel_auc


def ge_msking(protoflow, data_loader, args):
    p_list, b_list, r_list = [], [], []
    for mask_ratio in tqdm(np.arange(0.01, 0.08, 0.02).tolist()):
        randommask_generator = RandomMaskGenerator(
            args.window_size, num_masking_patches=int(196 * 0.2))
        blockmask_generator = BlockWiseMaskGenerator(
            args.window_size, num_masking_patches=int(196 * 0.15))
        pre_masks, img_labels, abnomalities = protoflow.predict(data_loader, args.class_name, mask_ratio)
        pres_list, rs_list, bs_list = [], [], []
        with torch.no_grad():
            for j, pre_mask in enumerate(pre_masks):
                pre_mask = torch.from_numpy(pre_mask).to(torch.long)
                pre_mask = pre_mask.unsqueeze(0)
                Prototype_mask = pre_mask
                prototype_mask = Prototype_mask.reshape(1, 196).to(torch.bool)
                pres_list.append(prototype_mask)
                block_mask = blockmask_generator().to(torch.long).unsqueeze(0)
                block_mask = block_mask.reshape(1, 196).to(torch.bool)
                bs_list.append(block_mask)
                random_mask = randommask_generator().to(torch.long).unsqueeze(0)
                random_mask = random_mask.reshape(1, 196).to(torch.bool)
                rs_list.append(random_mask)
            preds_mask = torch.cat(pres_list, dim=0)
            preds_mask = preds_mask.unsqueeze(1)
            p_list.append(preds_mask)
            rs_mask = torch.cat(rs_list, dim=0)
            rs_mask = rs_mask.unsqueeze(1)
            r_list.append(rs_mask)
            bs_mask = torch.cat(bs_list, dim=0)
            bs_mask = bs_mask.unsqueeze(1)
            b_list.append(bs_mask)
    p_batch = torch.cat(p_list, dim=1)
    r_batch = torch.cat(r_list, dim=1)
    b_batch = torch.cat(b_list, dim=1)
    return p_batch, r_batch, b_batch, abnomalities


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    model = prepare_model(args, args.model_path)
    print('Model loaded.')
    model.to(device)
    effmodel = get_feature_model().to(device)
    protoflow = ProtoFlow('weights/protoflow.pt')
    eplisons = 0.1
    OT_cos = SinkhornDistance(eps=eplisons, max_iter=1000, reduction=None, dis='cos', device=args.device)
    all_image_aucs, all_pixel_aucs = [], []
    for class_name in MVTEC.CLASS_NAMES:
        args.class_name = class_name
        image_auc, pixel_auc = evaluation(args, model, protoflow, effmodel, OT_cos)
        all_image_aucs.append(image_auc)
        all_pixel_aucs.append(pixel_auc)
    for i, class_name in enumerate(MVTEC.CLASS_NAMES):
        print(
            f'{class_name}: Image-AUC: {all_image_aucs[i] * 100:.3f}, Pixel-AUC: {all_pixel_aucs[i] * 100:.3f}')
    print('Mean Image-AUC: {:.3f}'.format(np.mean(all_image_aucs) * 100))
    print('Mean Pixel-AUC: {:.3f}'.format(np.mean(all_pixel_aucs) * 100))


def parse_args():
    parser = argparse.ArgumentParser(description='Mask AD:mvtec')

    parser.add_argument('--model_path', type=str,
                        default='', help='checkpoint path of model')
    parser.add_argument('--model', default='vit_base_patch16_224_8k_vocab', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--ref_data_path", type=str, default="")
    parser.add_argument('--class_name', default='bottle')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.3, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    parser.add_argument('--second_input_size', default=224, type=int,
                        help='images input size for discrete vae')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=4)
    parser.add_argument('--OT_w', type=float, default=0.6, help='ot score weight')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
