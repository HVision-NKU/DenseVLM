import random
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import re
import sys

class DenseVLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Load pre-trained embeddings from numpy files
        self.uvlm_embeddings = torch.from_numpy(np.load(args.uvlm_embed_path))
        self.pvlm_embeddings = torch.from_numpy(np.load(args.pvlm_embed_path))

        match = re.search(r'Thing(\d+)_STUFF', args.pvlm_embed_path)
        if match:
            self.thing = int(match.group(1)) - 1   # Threshold for distinguishing thing and stuff classes
            print(f"Thing threshold: {self.thing }")
        else:
            print("Failed to extract number from filename.")
            sys.exit(1)  # Exit with error code 1
        
    def __call__(self, batch, model, dist_P_VLM, dist_model, loss, device, cast_dtype, distributed, args):
        """
        Forward pass for dense visual-language model training with knowledge distillation.
        
        Args:
            batch: Input batch containing images, normalized boxes, and image crops
            model: 
            dist_P_VLM: 
            ... [other arguments]
        
        Returns:
            losses: Dictionary of computed losses
            batch_size: Number of samples in the batch
            temperature: Scaling factor from the model
        """
        # Handle distributed training setup
        if distributed:
            model = model.module
            dist_P_VLM = dist_P_VLM.module
        
        # Unpack batch and move to device
        images, normed_boxes, image_crops = batch
        del image_crops
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        normed_boxes = normed_boxes.to(device=device, dtype=cast_dtype, non_blocking=True)
        
        # Move embeddings to device
        self.uvlm_embeddings = self.uvlm_embeddings.to(device=device, dtype=cast_dtype, non_blocking=True)
        self.pvlm_embeddings = self.pvlm_embeddings.to(device=device, dtype=cast_dtype, non_blocking=True)

        # Handle multi-scale training with random resizing
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
            images_mul = F.interpolate(images, size=(tar_size, tar_size), mode='bilinear')

        # Extract valid region of interest boxes
        rois_list = []
        for bboxes_per_image in normed_boxes:
            valid = bboxes_per_image[:, -1] > 0.5
            rois_list.append(bboxes_per_image[valid, :4])

        # Compute P_VLM features and predictions with no gradient computation
        with torch.no_grad():
            pvlm_roi_features = dist_P_VLM.encode_pseudo_boxes(images, rois_list, normalize=False)
            pvlm_normed_features = F.normalize(pvlm_roi_features, dim=-1)

            # Compute similarity scores and filter by confidence
            scale_factor = 100
            pvlm_logits_image = scale_factor * pvlm_normed_features @ self.pvlm_embeddings.T
            
            pvlm_pre_pro, pvlm_pre_label = pvlm_logits_image.softmax(-1).max(-1)

     
            valid = pvlm_pre_pro > 0.3  # Confidence threshold
            
            pvlm_normed_features = pvlm_normed_features[valid, :]
            
            pvlm_logits_image = scale_factor * pvlm_normed_features @ self.pvlm_embeddings.T
            pvlm_pre_pro, pvlm_pre_label = pvlm_logits_image.softmax(-1).max(-1)
            index = pvlm_pre_label < self.thing   # Mask for thing classes

        # Compute student features for valid regions
        uvlm_roi_features = model.encode_pseudo_boxes(images, rois_list, normalize=False, extract_type=args.extract_type)
        uvlm_roi_features = uvlm_roi_features[valid, :]
        uvlm_normed_features = F.normalize(uvlm_roi_features, dim=-1)

        # Compute knowledge distillation loss
        if index.any():
            # Handle thing classes (pre_label < thing)
            pvlm_logits_image_thing = scale_factor * pvlm_normed_features[index] @ self.pvlm_embeddings[:self.thing ].T
            uvlm_logits_image_thing = scale_factor * uvlm_normed_features[index] @ self.uvlm_embeddings[:self.thing ].T

            loss_kl = F.kl_div(
                uvlm_logits_image_thing.log_softmax(dim=1),
                pvlm_logits_image_thing.softmax(dim=1),
                reduction='batchmean'
            )

            # Handle stuff classes (pre_label >= thing) if any
            if (~index).any():
                pvlm_logits_image = scale_factor * pvlm_normed_features[~index] @ self.pvlm_embeddings.T
                uvlm_logits_image = scale_factor * uvlm_normed_features[~index] @ self.uvlm_embeddings.T

                loss_kl += F.kl_div(
                    uvlm_logits_image.log_softmax(dim=1),
                    pvlm_logits_image.softmax(dim=1),
                    reduction='batchmean'
                )
            
        else:
            # All samples are stuff classes
            pvlm_logits_image = scale_factor * pvlm_normed_features @ self.pvlm_embeddings.T
            uvlm_logits_image = scale_factor * uvlm_normed_features @ self.uvlm_embeddings.T
            
            loss_kl = F.kl_div(
                uvlm_logits_image.log_softmax(dim=1),
                pvlm_logits_image.softmax(dim=1),
                reduction='batchmean'
            )

        losses = dict(loss_kl=loss_kl)
        return losses, len(images), model.logit_scale.exp()