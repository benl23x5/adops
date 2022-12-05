CFLAGS=-std=c99 -O3 -lm

OS=$(shell uname -s)
ifeq ($(OS),Darwin)
OPENCL_LDFLAGS?=-framework OpenCL
else
OPENCL_LDFLAGS?=-lOpenCL
endif

NAME=conv2d_dKrn

.PHONY: clean run

all: $(NAME)_bench $(NAME)_bench_cl

$(NAME).c $(NAME).h: ../../ops/$(NAME).fut
	futhark c --library \
		../../ops/$(NAME).fut \
		--entry-point=$(NAME)_flat \
		--entry-point=$(NAME)_impl_flat \
		-o $(NAME)

$(NAME)_cl.c $(NAME)_cl.h: ../../ops/$(NAME).fut
	futhark opencl --library \
		../../ops/$(NAME).fut \
		--entry-point=$(NAME)_flat \
		--entry-point=$(NAME)_impl_flat \
		-o $(NAME)_cl

$(NAME)_bench: $(NAME)_bench.c $(NAME).c $(NAME).h
	gcc \
		$(NAME)_bench.c $(NAME).c \
		-o $(NAME)_bench $(CFLAGS)

$(NAME)_bench_cl: $(NAME)_bench.c $(NAME)_cl.c $(NAME)_cl.h
	gcc \
		-DOPENCL=1 \
		$(NAME)_bench.c $(NAME)_cl.c \
		-o $(NAME)_bench_cl \
		$(CFLAGS) \
		$(OPENCL_LDFLAGS)

clean:
	rm -f\
	  $(NAME)_bench $(NAME).c $(NAME).h \
	  $(NAME)_bench_cl $(NAME)_cl.c $(NAME)_cl.h \
		*.json *.csv