CHECKPOINT=$1
GPU=$2
NAME=$3

torchrun --nproc_per_node $GPU -m training.main --batch-size=1 \
--model EVA02-CLIP-B-16 --pretrained eva --test-type ade_panoptic --train-data="" \
--val-data data/ADEChallengeData2016/ade20k_panoptic_val.json \
--embed-path metadata/ade_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy \
--val-image-root data/ADEChallengeData2016/images/validation \
--val-segm-root data/ADEChallengeData2016/ade20k_panoptic_val \
--cache-dir $CHECKPOINT --extract-type="v2" \
--name $NAME --downsample-factor 16 --det-image-size 512