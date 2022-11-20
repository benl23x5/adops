#!/bin/bash

SAMPLE=B-0500-0009

mkdir -p output
rm -f output/*
time cabal run adops stereo \
  data/flyingthings/${SAMPLE}-left.bmp \
  data/flyingthings/${SAMPLE}-right.bmp \
  data/stereo/params \
  output

convert output/disparity.bmp -scale 920x512 output/disparity-920x512.png
