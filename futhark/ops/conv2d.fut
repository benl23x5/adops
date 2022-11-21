

def tabulate_4d
    (nN: i64)(nC: i64)(nH: i64)(nW: i64)
    (f: i64 -> i64 -> i64 -> i64 -> f32)
 :  [nN][nC][nH][nW]f32
 = tabulate nN (\iN ->
    tabulate nC (\iC ->
     tabulate nH (\iH ->
      tabulate nW (\iW -> f iN iC iH iW))))


def slice4
    [nAi][nAc][nAh][nAw]
    (nKi: i64) (nKc: i64) (nKh: i64) (nKw: i64)
    (arrA:  [nAi][nAc][nAh][nAw]f32)
    (start: (i64, i64, i64, i64))
 :  ([nKi][nKc][nKh][nKw]f32)
 = arrA[
    start.0 : start.0 + nKi,
    start.1 : start.1 + nKc,
    start.2 : start.2 + nKh,
    start.3 : start.3 + nKw] :> [nKi][nKc][nKh][nKw]f32


def mmap4
    [nN][nC][nH][nW]
    (f: f32 -> f32 -> f32)
    (a: [nN][nC][nH][nW]f32)
    (b: [nN][nC][nH][nW]f32)
 :  [nN][nC][nH][nW]f32
 = map2 (\x1 y1 ->
   map2 (\x2 y2 ->
   map2 (\x3 y3 ->
   map2 (\x4 y4 -> f x4 y4) x3 y3) x2 y2) x1 y1) a b


def dot4
    [nN][nC][nH][nW]
    (arrA: [nN][nC][nH][nW]f32)
    (arrB: [nN][nC][nH][nW]f32)
 : f32
 = reduce (\x y -> x + y) 0 (flatten_4d (mmap4 (\x y -> x * y) arrA arrB))


def conv2d
    [nAi][nAc][nAh][nAw]
    [nBc][nKh][nKw]
    [nBh][nBw]
    (arrA: [nAi][nAc][nAh][nAw]f32)
    (arrK: [nBc][nAc][nKh][nKw]f32)
 : ([nAi][nBc][nBh][nBw]f32)
 = tabulate_4d nAi nBc nBh nBw (\iImg iCout iBh iBw ->
        let arrAt = slice4 1 nAc nKh nKw arrA (iImg,  0, iBh, iBw)
        let arrKt = slice4 1 nAc nKh nKw arrK (iCout, 0,   0,   0)
        in  dot4 arrAt arrKt)


def conv2d_dInp
    [nAi][nAc][nAh][nAw]
    [nBc][nKh][nKw]
    [nBh][nBw]
    (arrA: [nAi][nAc][nAh][nAw]f32)
    (arrK: [nBc][nAc][nKh][nKw]f32)
    (arrO: [nAi][nBc][nBh][nBw]f32)
 : ([nBc][nAc][nKh][nKw]f32)
 = vjp (conv2d arrA) arrK arrO

