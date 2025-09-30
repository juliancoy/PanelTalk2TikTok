nvidia-docker run \
  --privileged \
  -v $PWD:/app \
  -v "ifan-python-packages":"/usr/local/lib/python3.9/site-packages" \
  -w /app \
  --gpus=all \
  -it \
  --name IFAN \
  --rm \
  codeslake/ifan:CVPR2021 \
  /bin/zsh
