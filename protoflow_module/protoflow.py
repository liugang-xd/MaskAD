import os
import timm
import faiss
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

import FrEIA.framework as Ff
import FrEIA.modules as Fm
from datasets.test import MVTEC
from protoflow_module.modules import get_logp, positionalencoding2d
from protoflow_module.utils import embedding_concat

gamma = 0.0
theta = torch.nn.Sigmoid()
log_theta = torch.nn.LogSigmoid()
idx_to_class = {0: 'bottle', 1: 'cable', 2: 'capsule', 3: 'carpet',
                4: 'grid', 5: 'hazelnut', 6: 'leather', 7: 'metal_nut',
                8: 'pill', 9: 'screw', 10: 'tile', 11: 'toothbrush',
                12: 'transistor', 13: 'wood', 14: 'zipper'}
coreset_bank = {}
base_path = 'weights/prototypes'
for class_name in MVTEC.CLASS_NAMES:
    embedding_path = os.path.join(base_path, class_name)
    coreset = np.load(os.path.join(embedding_path, 'coreset.npy'))
    coreset_bank[class_name] = coreset


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2 * dims_in), nn.ReLU(), nn.Linear(2 * dims_in, dims_out))


def freia_cflow_head(n_feat, dim_c):
    coder = Ff.SequenceINN(n_feat)
    print('Condition NormFlow Coder: feature dim: ', n_feat)
    for k in range(8):
        coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(dim_c,), subnet_constructor=subnet_fc, affine_clamping=1.9,
                     global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def load_decoder_arch(dim_in, dim_c=128):
    decoder = freia_cflow_head(dim_in, dim_c)

    return decoder


def test_external_decoder(decoder, index, e_r, c_r, n_neighbors=1, size=32):
    C = e_r.shape[1]
    index, coreset = index
    e_r_np = e_r.cpu().numpy()

    _, idx = index.search(e_r_np, k=n_neighbors)
    index_feats = coreset[idx[:, 0]]
    index_feats = torch.from_numpy(index_feats).to(e_r.device)

    e_pair = e_r - index_feats
    z, log_jac_det = decoder(e_pair, [c_r, ])

    decoder_log_prob = get_logp(C, z, log_jac_det)
    log_prob = decoder_log_prob / C
    loss = -log_theta(log_prob).mean()
    log_prob = log_prob.reshape(size, size)

    return log_prob, loss


def generate_prototype(embeds, labels):

    protos = []
    for e, label in zip(embeds, labels):
        coreset = coreset_bank[idx_to_class[label.item()]]
        index = faiss.IndexFlatL2(e.shape[-1])
        index.add(coreset)

        e_np = e.cpu().numpy()

        _, idx = index.search(e_np, k=1)
        index_feats = coreset[idx[:, 0]]
        index_feats = torch.from_numpy(index_feats).to(e.device)

        p_r = index_feats
        protos.append(p_r)
    protos = torch.stack(protos, dim=0)

    return protos


class ProtoFlow(object):
    def __init__(self, path, device='cuda'):
        self.device = device
        self.condition_vec = 128

        encoder = timm.create_model('tf_efficientnet_b6', features_only=True,
                                    out_indices=(2, 3), pretrained=True)
        self.encoder = encoder.to(self.device).eval()
        pool_dims = self.encoder.feature_info.channels()
        decoder = load_decoder_arch(sum(pool_dims), self.condition_vec)
        self.decoder = decoder.to(self.device)

        state = torch.load(path)
        self.encoder.load_state_dict(state['encoder_state_dict'], strict=False)
        self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)

    def predict(self, data_loader, class_name, mask_ratio=0.3):
        coreset = coreset_bank[class_name]
        index = faiss.IndexFlatL2(coreset.shape[1])
        index.add(coreset)
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        index = (index, coreset)

        P = self.condition_vec
        decoder = self.decoder.eval()

        gt_label_list = list()
        gt_mask_list = list()
        e_logp_list = []
        with torch.no_grad():
            for (image, token_img, label, mask, img_path, img_type) in data_loader:
                gt_label_list.extend(label.numpy())
                gt_mask_list.extend(mask.numpy())

                image = image.to(self.device)
                outputs = self.encoder(image)

                embeddings = []
                for feature in outputs:
                    m = torch.nn.AvgPool2d(3, 1, 1)
                    embeddings.append(m(feature))
                embedding = embedding_concat(embeddings[0], embeddings[1])

                B, C, H, W = embedding.size()
                S = H * W
                E = B * S


                p = positionalencoding2d(P, H, W).to(self.device).unsqueeze(0).repeat(B, 1, 1, 1)

                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)

                e_r = embedding.reshape(B, C, S).transpose(1, 2).reshape(E, C)
                e_r = e_r.contiguous()


                external_logp, _ = test_external_decoder(decoder, index, e_r, c_r, size=H)

                e_logp_list.append(external_logp)

        logp = torch.stack(e_logp_list, dim=0)
        logp -= torch.max(logp)
        prob = torch.exp(logp)
        rank = prob.max() - prob
        rank = F.interpolate(rank.unsqueeze(1), size=14, mode='bilinear', align_corners=True).squeeze()

        prob = F.interpolate(prob.unsqueeze(1), size=224, mode='bilinear', align_corners=True).squeeze().cpu().numpy()
        abnomality = prob.max() - prob

        img_scores = abnomality.reshape(abnomality.shape[0], -1).max(axis=1)
        gt_list = np.array(gt_label_list)
        precision, recall, thresholds = precision_recall_curve(gt_list.flatten(), img_scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        image_threshold = thresholds[np.argmax(f1)]
        img_labels = img_scores > image_threshold

        rank = rank.reshape(-1, 196)
        thres = torch.topk(rank, int(196 * mask_ratio), dim=-1)[0][:, -1]
        thres = thres.unsqueeze(-1)
        thres = thres.expand([thres.shape[0], rank.shape[-1]])
        pre_mask = rank >= thres

        pre_mask = pre_mask.cpu().numpy()

        return pre_mask, img_labels, abnomality
