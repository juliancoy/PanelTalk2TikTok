CUDA_VISIBLE_DEVICES=0 \
python IFAN/run.py \
  --mode IFAN_44 \
  --network IFAN \
  --config config_IFAN_44 \
  --data folder \
  --ckpt_abs_name IFAN/ckpt/IFAN_44.pytorch \
  --data_offset "$1"
