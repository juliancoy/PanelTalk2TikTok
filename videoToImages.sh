#!/bin/bash

video="$1"
outdir="$(basename "$video" | sed 's/\.[^.]*$//')"

mkdir -p "$outdir"
#ffmpeg -i "$video" "$outdir/%05d.jpg"
ffmpeg -i "$video" "$outdir/%05d.png" # Quality is much better when PNG is fed
