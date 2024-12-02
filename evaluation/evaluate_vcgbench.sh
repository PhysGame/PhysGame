export PYTHONPATH="./:$PYTHONPATH"
torchrun --nproc_per_node=8 --master_port=33099 physvlm/test/vcgbench/vcgbench_infer_mp.py \
    --cfg-path config/physvlm_dpo_training.yaml \
    --ckpt-path ./ckpt/checkpoint_dpo \
    --output_dir output/VCG/result \
    --num-frames 32 \
    