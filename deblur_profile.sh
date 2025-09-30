CUDA_VISIBLE_DEVICES=0 \
python -m cProfile -o output.prof IFAN/run.py \
  --mode IFAN \
  --network IFAN \
  --config config_IFAN \
  --data random \
  --ckpt_abs_name IFAN/ckpt/IFAN.pytorch \
  --data_offset ./data_offset \
  --output_offset ./output
