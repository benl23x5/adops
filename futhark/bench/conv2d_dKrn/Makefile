CFLAGS=-std=c99 -O3 -lm
NAME=conv2d_dKrn

.PHONY: clean run

all: $(NAME)_bench

$(NAME).c $(NAME).h: ../../ops/$(NAME).fut
	futhark c --library ../../ops/$(NAME).fut --entry-point=$(NAME) -o $(NAME)

$(NAME)_bench: $(NAME)_bench.c $(NAME).c $(NAME).h
	gcc $(NAME)_bench.c $(NAME).c -o $(NAME)_bench $(CFLAGS)

clean:
	rm -f $(NAME)_bench $(NAME).c $(NAME).h $(NAME).json