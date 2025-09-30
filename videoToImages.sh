#!/bin/bash

video="$1"
outdir="$(basename "$video" | sed 's/\.[^.]*$//')"

mkdir -p "$outdir"
ffmpeg -i "$video" "$outdir/%05d.jpg"
