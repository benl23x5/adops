CFLAGS=-std=c99 -O3 -lm
K=conv2d_dKrn

.PHONY: clean run

all: $(K)_bench

$(K).c $(K).h: ../ops/$(K).fut
	futhark c --library ../ops/$(K).fut --entry-point=$(K) -o $(K)

$(K)_bench: $(K)_bench.c $(K).c $(K).h
	gcc $(K)_bench.c $(K).c -o $(K)_bench $(CFLAGS)

clean:
	rm -f $(K) $(K).c $(K).h $(K).json