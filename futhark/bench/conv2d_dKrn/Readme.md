# Benchmark Futhark AD vs. Custom Implementation for conv2d_dKrn

## Unpadded Slices

There is some disparencies between running our Futhark conv2d as an executable
and as a library.

Compile `conv2d_dKrn.fut` as an executable with baked-in input sizes: works.

- `slice4` doesn't do any padding and can go out of bounds, unlike in adops.
- `slice4` is used in both `conv2d_dKrn` and `conv2d_dKrn_impl`.
- `test1` compares the two derivatives.
- Compile with:
  ```
  futhark c conv2d_dKrn.fut --entry-point=test1 --executable -o dump/conv2d_test
  ```
- Run with:
  ```
  echo | ./dump/conv2d_test --entry-point=test1
  true
  ```

Compile `conv2d_dKrn.fut` as a library and run with a C wrapper: failed (with the same input sizes.)

```
$ make && ./conv2d_dKrn_bench
futhark c --library ../../ops/conv2d_dKrn.fut --entry-point=conv2d_dKrn -o conv2d_dKrn
gcc conv2d_dKrn_bench.c conv2d_dKrn.c -o conv2d_dKrn_bench -std=c99 -O3 -lm
Error: Index [0:1, 0:2, 0:3, 30:33] out of bounds for array of shape [1][2][32][32].

Backtrace:
-> #0  ../../ops/conv2d_dKrn.fut:18:4-22:28
   #1  ../../ops/conv2d_dKrn.fut:63:1-71:30
```

## Padded Slices

Replaced `slice4` with `slicez4` -- a very slow padded slice with a conditional
on every index. As it is used in both the AD `dKrn` and the custom
implementation, the benchmark is still valid.