import jittor as jt
import jittor.nn as nn

class ClipLoss(nn.Module):
    def __init__(self, cache_labels=False):
        super().__init__()
        self.cache_labels = cache_labels
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, num_logits):
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits:
            labels = jt.arange(num_logits, dtype=jt.int32)
            if self.cache_labels:
                self.labels['default'] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels['default']
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        return logits_per_image, logits_per_text

    def execute(self, image_features, text_features, logit_scale, output_dict=False):
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        labels = self.get_ground_truth(logits_per_image.shape[0])
        total_loss = (
            nn.cross_entropy_loss(logits_per_image, labels) +
            nn.cross_entropy_loss(logits_per_text, labels)
        ) / 2
        return {"contrastive_loss": total_loss} if output_dict else total_loss

class CoCaLoss(ClipLoss):
    def __init__(self, caption_loss_weight, clip_loss_weight, pad_id=0, cache_labels=False):
        super().__init__(cache_labels=cache_labels)
        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def execute(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        clip_loss = super().execute(image_features, text_features, logit_scale)
        clip_loss = self.clip_loss_weight * clip_loss
        caption_loss = self.caption_loss(logits.permute(0, 2, 1), labels)
        caption_loss = caption_loss * self.caption_loss_weight
        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}
        return clip_loss, caption_loss

class DistillClipLoss(ClipLoss):
    def dist_loss(self, teacher_logits, student_logits):
        loss = nn.kl_div_loss(nn.log_softmax(student_logits, dim=1), nn.softmax(teacher_logits, dim=1), reduction='batchmean')
        return loss

    def execute(self, image_features, text_features, logit_scale, dist_image_features, dist_text_features, dist_logit_scale, output_dict=False):
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        dist_logits_per_image, dist_logits_per_text = self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)
        labels = self.get_ground_truth(logits_per_image.shape[0])
        contrastive_loss = (
            nn.cross_entropy_loss(logits_per_image, labels) +
            nn.cross_entropy_loss(logits_per_text, labels)
        ) / 2
        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2
        if output_dict:
            return {"contrastive_loss": contrastive_loss, "loss_kl": distill_loss}
        return contrastive_loss, distill_loss
