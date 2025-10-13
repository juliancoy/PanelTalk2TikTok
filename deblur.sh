CUDA_VISIBLE_DEVICES=0 \
python IFAN/run.py \
  --mode IFAN \
  --network IFAN \
  --config config_IFAN \
  --data folder \
  --ckpt_abs_name IFAN/ckpt/IFAN.pytorch \
  --data_offset "$1" \
  --batch_size "${2:-1}"
