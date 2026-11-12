from math import sqrt

fn rmsnorm_row[WIDTH: Int](
    x: UnsafePointer[DType.float32],
    w: UnsafePointer[DType.float32],
    out: UnsafePointer[DType.float32],
    base: Int,
    eps: Float32,
):
    var sum_sq: Float32 = 0.0
    for index in range(WIDTH):
        let value = x.load(base + index)
        sum_sq += value * value
    let inverse = Float32(1.0) / sqrt(sum_sq / Float32(WIDTH) + eps)
    for index in range(WIDTH):
        out.store(base + index, x.load(base + index) * inverse * w.load(index))


fn rmsnorm_rows[WIDTH: Int](
    x: UnsafePointer[DType.float32],
    w: UnsafePointer[DType.float32],
    out: UnsafePointer[DType.float32],
    rows: Int,
    eps: Float32,
):
    for row in range(rows):
        rmsnorm_row[WIDTH](x, w, out, row * WIDTH, eps)


fn rmsnorm_chunked[WIDTH: Int](
    x: UnsafePointer[DType.float32],
    w: UnsafePointer[DType.float32],
    out: UnsafePointer[DType.float32],
    rows: Int,
    eps: Float32,
    threads: Int,
):
    let per = (rows + threads - 1) // threads
    for chunk in range(threads):
        let begin = chunk * per
        if begin >= rows:
            break
        let end = min(begin + per, rows)
        for row in range(begin, end):
            rmsnorm_row[WIDTH](x, w, out, row * WIDTH, eps)
