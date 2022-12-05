
set datafile separator ','
set key autotitle columnhead

set terminal png size 800,600

set mxtics 10
set xrange[1:100000]
set grid xtics ytics mxtics mytics
set logscale

CSV_AD   = "conv2d_dKrn_ad_cl.csv"
CSV_IMPL = "conv2d_dKrn_impl_cl.csv"

# ----------
set title "\
conv2d-dKrn\n\
futhark-ad vjp vs build-index custom vjp\n\
OpenCL"
set output "conv2d_dKrn_cl.png"
set xlabel "number of convolution filters"
set ylabel "time (ms)"
plot    CSV_AD   using 5:9 \
         with linespoints title "conv2d-dKrn", \
        CSV_IMPL using 5:9  \
         with linespoints title "conv2d-dKrn-impl"

