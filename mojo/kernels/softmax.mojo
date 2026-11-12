from math import exp

fn softmax_row[WIDTH: Int](x: UnsafePointer[DType.float32], out: UnsafePointer[DType.float32], base: Int):
    var maximum: Float32 = x.load(base)
    for index in range(1, WIDTH):
        let value = x.load(base + index)
        if value > maximum:
            maximum = value

    var total: Float32 = 0.0
    for index in range(WIDTH):
        let exponent = exp(x.load(base + index) - maximum)
        out.store(base + index, exponent)
        total += exponent

    let inverse = Float32(1.0) / total
    for index in range(WIDTH):
        out.store(base + index, out.load(base + index) * inverse)


fn softmax_rows[WIDTH: Int](
    x: UnsafePointer[DType.float32],
    out: UnsafePointer[DType.float32],
    rows: Int,
):
    for row in range(rows):
        softmax_row[WIDTH](x, out, row * WIDTH)
