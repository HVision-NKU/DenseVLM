import logging
import jittor as jt
import jittor.nn as nn
import numpy as np
from tqdm import tqdm
from .distributed import is_master
from .precision import get_autocast
from jittor.dataset import DataLoader

def l2_normalize(x, dim=-1, eps=1e-8):
    norm = jt.sqrt((x ** 2).sum(dim=dim, keepdims=True))
    return x / (norm + eps)



def run(model, dataloader, args):
    # dataloader = DataLoader(dataloader, batch_size=getattr(args, 'batch_size', 1), shuffle=False, num_workers=0)
    cls_embeddings = getattr(dataloader, 'embeddings', None)
    if cls_embeddings is None and hasattr(dataloader, 'dataset'):
        cls_embeddings = getattr(dataloader.dataset, 'embeddings', None)
    cls_embeddings = l2_normalize(jt.array(cls_embeddings).float32(), dim=-1)
    autocast = get_autocast(args.precision)
    with jt.no_grad():
        correct_rois = []
        correct_maskpool = []
        correct_crops = []
        similarity_crops = []
        similarity_rois = []
        similarity_maskpool = []
        all_box_sizes = []
        all_is_thing = []
        all_cls_labels = []
        for images, bboxes, image_crops, gt_masks, masked_image_crops in tqdm(dataloader):
            images = jt.array(images)
            bboxes = jt.array(bboxes)
            image_crops = jt.array(image_crops)
            masked_image_crops = jt.array(masked_image_crops)
            gt_masks = jt.array(gt_masks)
            image_crops_list = []
            gt_masks_list = []
            cls_labels = []
            rois = []
            box_sizes = []
            is_thing = []
            
            for bboxes_per_image, crops_per_image, gt_mask, masked_crops_per_image in zip(bboxes, image_crops, gt_masks, masked_image_crops):
                # 确保比较值与张量类型一致
                threshold = jt.array(0.5, dtype=bboxes_per_image.dtype)
                valid = bboxes_per_image[:, 5] > threshold
                
                
            for bboxes_per_image, crops_per_image, gt_mask, masked_crops_per_image in zip(bboxes, image_crops, gt_masks, masked_image_crops):
                valid = bboxes_per_image[:, 5] > 0.5
                # 先忽略 box 分数，计算 mask 分数
                # rois.append(bboxes_per_image[valid, :4])
                breakpoint()
                #* ???
                cls_labels.append(bboxes_per_image[valid, 4])
                
                image_crops_list.append(crops_per_image[valid])
                gt_masks_list.append(gt_mask[valid])
                box_sizes.append(bboxes_per_image[valid, 6])
                is_thing.append(bboxes_per_image[valid, 7])
                
            cls_labels = jt.concat(cls_labels, dim=0).int32()
            if cls_labels.shape[0] == 0:
                continue
            
            
            image_crops = jt.concat(image_crops_list, dim=0)
            box_sizes = jt.concat(box_sizes, dim=0).float32()
            is_thing = jt.concat(is_thing, dim=0)
            all_box_sizes.append(box_sizes)
            all_is_thing.append(is_thing)
            module = model
            
            # roi_extractor = module.encode_pseudo_boxes #! 
            # roi_features = roi_extractor(images, rois, normalize=True, extract_type=args.extract_type)
            
            mask_pooler = module.encode_masks #!
            # images.shape
            # [1,3,512,512,]
            # gt_masks_list[0].shape
            # [7,32,32,]
            maskpool_features = mask_pooler(images, gt_masks_list, normalize=True, mask_attn=args.extract_type == "v1")
            
            #* 此处调整
            # if args.image_ave_pool:
            #     feature_map = module.visual.encode_dense(image_crops, keep_shape=True)
            #     crop_features = jt.mean(feature_map, dims=[-2, -1])
            #     crop_features = l2_normalize(crop_features, dim=-1)
            # else:
            
            crop_features = module.encode_image(image_crops, normalize=True)
            
            # roi_logits = roi_features @ cls_embeddings.transpose(0, 1)
            crop_logits = crop_features @ cls_embeddings.transpose(0, 1)
            maskpool_logits = maskpool_features @ cls_embeddings.transpose(0, 1)
            
            # roi_top5_inds = jt.argsort(roi_logits, dim=1)[:, -5:]
            _, crop_top5_inds = jt.argsort(crop_logits, dim=1)
            crop_top5_inds = crop_top5_inds[:, -5:]
            _, maskpool_top5_inds = jt.argsort(maskpool_logits, dim=1)
            maskpool_top5_inds = maskpool_top5_inds[:, -5:]

            # correct_rois.append((roi_top5_inds == cls_labels.unsqueeze(1)).int32())
            correct_crops.append((crop_top5_inds.int32() == cls_labels.unsqueeze(1)).int32())
            correct_maskpool.append((maskpool_top5_inds.int32() == cls_labels.unsqueeze(1)).int32())
            # similarity_rois.append(jt.gather(roi_logits, 1, cls_labels.unsqueeze(1))[:, 0])
            similarity_crops.append(jt.gather(crop_logits, 1, cls_labels.unsqueeze(1))[:, 0])
            similarity_maskpool.append(jt.gather(maskpool_logits, 1, cls_labels.unsqueeze(1))[:, 0])
            all_cls_labels.append(cls_labels)
            
        
        correct_rois = jt.concat(correct_rois, dim=0).float32()
        correct_crops = jt.concat(correct_crops, dim=0).float32()
        correct_maskpool = jt.concat(correct_maskpool, dim=0).float32()
        similarity_rois = jt.concat(similarity_rois, dim=0).float32()
        similarity_crops = jt.concat(similarity_crops, dim=0).float32()
        similarity_maskpool = jt.concat(similarity_maskpool, dim=0).float32()
        all_box_sizes = jt.concat(all_box_sizes, dim=0)
        all_is_thing = jt.concat(all_is_thing, dim=0)
        all_cls_labels = jt.concat(all_cls_labels, dim=0)
    return correct_rois, correct_crops, correct_maskpool, \
        similarity_rois, similarity_crops, similarity_maskpool, \
        all_box_sizes, all_is_thing, all_cls_labels

def macc_with_is_thing(correct_matrix, is_thing, all_cls_labels, prefix):
    def _macc(corrects, cls_labels):
        min_id = int(jt.min(cls_labels).item())
        max_id = int(jt.max(cls_labels).item())
        cand_labels = list(range(min_id, max_id+1))
        acc_per_cls = []
        for lb in cand_labels:
            corrects_per_cls = corrects[cls_labels == lb]
            if corrects_per_cls.shape[0] == 0:
                continue
            acc_per_cls.append(float(jt.mean(corrects_per_cls)))
        return sum(acc_per_cls) / len(acc_per_cls) if acc_per_cls else 0.0
    results = {}
    thing_correct_matrix = correct_matrix[is_thing > 0]
    stuff_correct_matrix = correct_matrix[is_thing < 1]
    thing_cls_labels = all_cls_labels[is_thing > 0].int32()
    stuff_cls_labels = all_cls_labels[is_thing < 1].int32()
    thing_top1_acc = _macc(thing_correct_matrix[:, 0], thing_cls_labels)
    thing_top5_acc = _macc(jt.sum(thing_correct_matrix, dims=-1), thing_cls_labels)
    stuff_top1_acc = _macc(stuff_correct_matrix[:, 0], stuff_cls_labels)
    stuff_top5_acc = _macc(jt.sum(stuff_correct_matrix, dims=-1), stuff_cls_labels)
    results[f'{prefix}.thing.macc1'] = thing_top1_acc
    results[f'{prefix}.thing.macc5'] = thing_top5_acc
    results[f'{prefix}.stuff.macc1'] = stuff_top1_acc
    results[f'{prefix}.stuff.macc5'] = stuff_top5_acc
    return results

def zero_shot_eval(model, data, epoch, args):
    if 'val' not in data:
        return {}
    if getattr(args, 'zeroshot_frequency', 1) == 0:
        return {}
    if (epoch % getattr(args, 'zeroshot_frequency', 1)) != 0 and epoch != getattr(args, 'epochs', 0):
        return {}
    logging.info('Region classifier')
    results = {}
    correct_rois, correct_crops, correct_maskpool, \
        similarity_rois, similarity_crops, similarity_maskpool, \
        all_box_sizes, all_is_thing, all_cls_labels = run(model, data['val'].dataloader, args)
    results.update(macc_with_is_thing(correct_rois, all_is_thing, all_cls_labels, 'rois'))
    results.update(macc_with_is_thing(correct_crops, all_is_thing, all_cls_labels, 'crops'))
    results.update(macc_with_is_thing(correct_maskpool, all_is_thing, all_cls_labels, 'maskpool'))
    return results

    
