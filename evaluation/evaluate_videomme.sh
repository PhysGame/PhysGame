export PYTHONPATH="./:$PYTHONPATH"
torchrun --nproc_per_node=8 --master_port=33098 physvlm/test/video_mme/videomme_infer.py \
    --cfg-path config/physvlm_dpo_training.yaml \
    --ckpt-path ./ckpt/checkpoint_dpo \
    --output_dir output/Videomme/ \
    --output_name Test \
    --num-frames 0 \
    --llava \
    --use_subtitles \