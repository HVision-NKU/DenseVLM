CHECKPOINT=$1
NAME=$2

# export debug=1
# export gdb_attach=1

python3 -m training.main --batch-size=1 \
--model ViT-B-16 --pretrained openai --test-type ade_panoptic --train-data="" \
--val-data data/ADEChallengeData2016/ade20k_panoptic_val.json \
--embed-path metadata/ade_panoptic_clip_hand_craft_openai_ViTB16.npy \
--val-image-root data/ADEChallengeData2016/images/validation \
--val-segm-root data/ADEChallengeData2016/ade20k_panoptic_val \
--cache-dir $CHECKPOINT --extract-type="v2" \
--name $NAME --downsample-factor 16 --det-image-size 512