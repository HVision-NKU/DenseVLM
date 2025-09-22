# densevlm.py (Jittor version)

import re
import sys
import random
import numpy as np
import jittor as jt
from jittor import nn

def kl_divergence(input_log_prob, target_prob):
    # 计算 KL 散度 KL(Q||P)
    # input_log_prob: log P，形状 [batch, num_classes]
    # target_prob: Q，形状 [batch, num_classes]
    return (target_prob * (jt.log(target_prob + 1e-8) - input_log_prob)).sum(dim=1).mean()


class DenseVLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Load pre-trained embeddings from numpy files
        self.uvlm_embeddings = jt.array(np.load(args.uvlm_embed_path)).stop_grad()
        self.pvlm_embeddings = jt.array(np.load(args.pvlm_embed_path)).stop_grad()

        match = re.search(r'Thing(\d+)_STUFF', args.pvlm_embed_path)
        if match:
            self.thing = int(match.group(1)) - 1
            print(f"Thing threshold: {self.thing}")
        else:
            print("Failed to extract number from filename.")
            sys.exit(1)

    def execute(self, batch, model, dist_P_VLM, dist_model=None, loss=None, device=None, cast_dtype="float32", distributed=False, args=None):
        if distributed:
            model = model.module
            dist_P_VLM = dist_P_VLM.module

        images, normed_boxes, image_crops = batch
        del image_crops
        images = jt.cast(images, cast_dtype)
        normed_boxes = jt.cast(normed_boxes, cast_dtype)

        self.uvlm_embeddings = self.uvlm_embeddings.cast(cast_dtype).stop_grad()
        self.pvlm_embeddings = self.pvlm_embeddings.cast(cast_dtype).stop_grad()

        if args.multiscale:
            cur_h, cur_w = images.shape[2:]
            assert cur_h == cur_w
            if cur_h == 1024:
                tar_sizes = [320, 640, 896, 1024]
            elif cur_h == 896:
                tar_sizes = [336, 448, 672, 896]
            elif cur_h == 512:
                tar_sizes = [320, 512, 672, 896]
            else:
                raise NotImplementedError
            tar_size = random.choice(tar_sizes)
            images = nn.interpolate(images, size=[tar_size, tar_size], mode='bilinear')

        rois_list = []
        for bboxes_per_image in normed_boxes:
            valid = bboxes_per_image[:, -1] > 0.5
            rois_list.append(bboxes_per_image[valid, :4])

        with jt.no_grad():
            pvlm_roi_features = dist_P_VLM.encode_pseudo_boxes(images, rois_list, normalize=False)
            pvlm_normed_features = pvlm_roi_features / (pvlm_roi_features.norm(dim=-1, keepdims=True) + 1e-6)

            scale_factor = 100.0
            pvlm_logits_image = scale_factor * (pvlm_normed_features @ self.pvlm_embeddings.transpose(0, 1))
            pvlm_softmax = jt.nn.softmax(pvlm_logits_image, dim=-1)
            # print("pvlm_softmax =", pvlm_softmax)
            # print("type(pvlm_softmax):", type(pvlm_softmax))
            # print("pvlm_softmax.shape:", pvlm_softmax.shape)

            pvlm_pre_label = jt.argmax(pvlm_softmax, dim=-1)[1]
            # print("pvlm_pre_label =", pvlm_pre_label)
            # print("type(pvlm_pre_label):", type(pvlm_pre_label))

            pvlm_pre_pro = jt.max(pvlm_softmax, dim=-1)

            valid = pvlm_pre_pro > 0.3
            pvlm_normed_features = pvlm_normed_features[valid, :]
            pvlm_logits_image = scale_factor * (pvlm_normed_features @ self.pvlm_embeddings.transpose(0, 1))
            pvlm_pre_label = jt.argmax(pvlm_logits_image, dim=-1)[1]
            
            print(type(pvlm_pre_label))
            index = pvlm_pre_label < self.thing

        uvlm_roi_features = model.encode_pseudo_boxes(images, rois_list, normalize=False, extract_type=args.extract_type)
        uvlm_roi_features = uvlm_roi_features[valid, :]
        uvlm_normed_features = uvlm_roi_features / (uvlm_roi_features.norm(dim=-1, keepdims=True) + 1e-6)

        if jt.sum(index) > 0:
            pvlm_logits_image_thing = scale_factor * (pvlm_normed_features[index] @ self.pvlm_embeddings[:self.thing].transpose(0, 1))
            uvlm_logits_image_thing = scale_factor * (uvlm_normed_features[index] @ self.uvlm_embeddings[:self.thing].transpose(0, 1))
            loss_kl = kl_divergence(
                jt.nn.log_softmax(uvlm_logits_image_thing, dim=1),
                jt.nn.softmax(pvlm_logits_image_thing, dim=1),
                )
            index_neg = jt.logical_not(index)
        if jt.sum(index_neg) > 0:
            pvlm_logits_image_stuff = scale_factor * (pvlm_normed_features[index_neg] @ self.pvlm_embeddings.transpose(0, 1))
            uvlm_logits_image_stuff = scale_factor * (uvlm_normed_features[index_neg] @ self.uvlm_embeddings.transpose(0, 1))
            loss_kl += kl_divergence(
                jt.nn.log_softmax(uvlm_logits_image_stuff, dim=1),
                jt.nn.softmax(pvlm_logits_image_stuff, dim=1),
                )
        else:
            pvlm_logits_image = scale_factor * (pvlm_normed_features @ self.pvlm_embeddings.transpose(0, 1))
            uvlm_logits_image = scale_factor * (uvlm_normed_features @ self.uvlm_embeddings.transpose(0, 1))
            loss_kl = kl_divergence(
                jt.nn.log_softmax(uvlm_logits_image, dim=1),
                jt.nn.softmax(pvlm_logits_image, dim=1),
                )

        losses = dict(loss_kl=loss_kl)
        return losses, images.shape[0], jt.exp(model.logit_scale)
