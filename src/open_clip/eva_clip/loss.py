import math
import jittor as jt
import jittor.nn as nn

class ClipLoss(nn.Module):
    def __init__(self, smoothing=0.):
        super().__init__()
        self.smoothing = smoothing
    def execute(self, image_features, text_features, logit_scale=1.):
        logits_per_image = logit_scale * image_features @ text_features.transpose(0, 1)
        logits_per_text = logit_scale * text_features @ image_features.transpose(0, 1)
        num_logits = logits_per_image.shape[0]
        labels = jt.arange(num_logits)
        if self.smoothing > 0:
            # label smoothing cross entropy
            loss_i = nn.cross_entropy_loss(logits_per_image, labels, reduction='mean', label_smooth=self.smoothing)
            loss_t = nn.cross_entropy_loss(logits_per_text, labels, reduction='mean', label_smooth=self.smoothing)
            total_loss = (loss_i + loss_t) / 2
        else:
            loss_i = nn.cross_entropy_loss(logits_per_image, labels, reduction='mean')
            loss_t = nn.cross_entropy_loss(logits_per_text, labels, reduction='mean')
            total_loss = (loss_i + loss_t) / 2
        i2t_acc = (jt.argmax(logits_per_image, dim=-1) == labels).sum() / len(logits_per_image)
        t2i_acc = (jt.argmax(logits_per_text, dim=-1) == labels).sum() / len(logits_per_text)
        acc = {"i2t": i2t_acc, "t2i": t2i_acc}
        return total_loss, acc