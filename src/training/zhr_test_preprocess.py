import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime

import numpy as np
import jittor as jt
from jittor import nn, optim

from open_clip import create_model_and_transforms, get_tokenizer, create_model

from training.data import get_data
from training.distributed import is_master, init_distributed_device, broadcast_object
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from training.train import train_one_epoch, evaluate, student_teacher_ensemble
from training.file_utils import pt_load
from training.densevlm import DenseVLM


# Name for the most recent checkpoint file
LATEST_CHECKPOINT_NAME = "epoch_latest.pt"
#* 修改3
if not hasattr(nn, "QuickGELU"):
    class QuickGELU(nn.Module):
        def execute(self, x):
            return x * nn.sigmoid(1.702 * x)
    nn.QuickGELU = QuickGELU
    
def random_seed(seed=42, rank=0):
    jt.set_global_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def natural_key(string_):
    """
    Sorts strings containing numbers in a natural order (e.g., file_9.pt, file_10.pt).
    See http://www.codinghorror.com/blog/archives/001018.html
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
    """
    Finds the path to the latest checkpoint file in a given directory.
    Supports local and remote (AWS S3) paths.
    """
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    """
    Main function to orchestrate model training and evaluation.
    """
    args = parse_args(args)
    jt.flags.use_cuda = 1


    # Initialize distributed device environment
    device = init_distributed_device(args) 

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None

    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)


    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level) # This might still be None for non-master ranks, handle in setup_logging if needed
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]

    random_seed(args.seed, args.rank)

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        output_dict=True,
        cache_dir=args.cache_dir,
        det_image_size=args.det_image_size,
        dataset_type=args.dataset_type,
    )
    args.input_size = model.visual.image_size
    
    dist_model = None
    dist_P_VLM = None
    


            
    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats,
        )
    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data:
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            # if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
            #     sd = {k[len('module.'):]: v for k, v in sd.items()}
            sd = {f'module.{k}': v for k, v in sd.items()}

            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
    

    #! 此处产生不一致,推测是 preprocess 和 preprocessval 不一致
    print("生成测试图像")
    from PIL import Image, ImageDraw
    # 创建 RGB 测试图像
    def create_test_image(path="test_image.jpg", size=(2048,2048)):
        img = Image.new("RGB", size, color=(128,128,128))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 200, 200], outline="red", width=5)
        draw.ellipse([100, 100, 150, 150], fill="blue")
        img.save(path)
        print(f"Test image saved to {path}")
        return img


    def apply_transform(transforms_list, image):
        """支持单个 Compose 或 list of Compose"""
        if not isinstance(transforms_list, (list, tuple)):
            transforms_list = [transforms_list]
        outputs = []
        for t in transforms_list:
            outputs.append(t(image))
        return outputs
    # 生成图片
    img = create_test_image()
    jt_img = apply_transform(preprocess_val, img)

    print(jt_img)
    for i, t in enumerate(jt_img):
        print(f"Tensor {i}: shape={t.shape}, dtype={t.dtype}, min={t.min().item()}, max={t.max().item()}")

    # # initialize datasets
    # data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=get_tokenizer(args.model))
    # assert len(data), 'At least one train or eval dataset must be specified.'


if __name__ == "__main__":
    main(sys.argv[1:])