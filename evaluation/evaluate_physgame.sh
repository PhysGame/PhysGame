export PYTHONPATH="./:$PYTHONPATH"
torchrun --nproc_per_node=8 --master_port=33099 physvlm/test/PhysGame_bench/PhysGame_bench_infer_mp.py \
    --cfg-path config/physvlm_dpo_training.yaml \
    --ckpt-path ./ckpt/checkpoint_dpo \
    --output_dir output/PhysGame/ \
    --output_name GamePyhsics_Test \
    --data_anno /path/to/PhysGame_880_annotation.json \
    --video_dir /path/to/videos \
    --num-frames 32 \
    --ask_simple \
    --llava \
    --system_llm \