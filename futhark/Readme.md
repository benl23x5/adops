# AD Ops in Futhark

Try to write deep learning style operators in Futhark. Want convolutions and
some unusal algorithms such as disparity cost volume.

The style is combinators based (input-oriented). The compiler has intrinsics for
map, reduce, gather, scatter, etc.

There is a deep learning library for Futhark but it doesn't use Futhark's
experimental AD support. It does convolutions with im2col/gemm.

Resources:
- Examples: https://futhark-lang.org/examples.html
- Prelude Index: https://futhark-lang.org/docs/prelude/doc-index.html
- Deep learning library:
  - https://github.com/HnimNart/deeplearning
  - https://futhark-lang.org/student-projects/duc-bsc-thesis.pdf
  - https://elsman.com/pdf/fhpnc19.pdf
