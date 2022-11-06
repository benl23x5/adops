
# AD Ops

Demo of array comprehension style definition of neural network operators.

## Stereo Disparity
```
$ mkdir output
$ cabal run adops stereo \
    data/flyingthings/A-0022-0015-left.bmp \
    data/flyingthings/A-0022-0015-right.bmp \
    output
```

## Sobel Edge Detection
```
$ mkdir output
$ cabal run adops sobel \
    data/test/butterfly.bmp \
    output
```

## Gradient Ascent
```
$ mkdir output
$ cabal run adops ascent \
    data/test/butterfly.bmp \
    output
```