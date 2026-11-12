from sys import argv, exit
from time import perf_counter

from kernels.rmsnorm import rmsnorm_rows
from rawio import read_f32, write_f32


fn arg_text(name: String, default: String) -> String:
    var index = 1
    while index < len(argv) - 1:
        if argv[index] == name:
            return argv[index + 1]
        index += 1
    return default


fn arg_int(name: String, default: Int) -> Int:
    let raw = arg_text(name, "")
    if raw == "":
        return default
    return Int(raw)


fn arg_float(name: String, default: Float64) -> Float64:
    let raw = arg_text(name, "")
    if raw == "":
        return default
    return Float64(raw)


fn sort_samples(values: UnsafePointer[DType.float64], count: Int):
    for index in range(1, count):
        let key = values.load(index)
        var scan = index - 1
        while scan >= 0 and values.load(scan) > key:
            values.store(scan + 1, values.load(scan))
            scan -= 1
        values.store(scan + 1, key)


fn dispatch[WIDTH: Int](
    x: UnsafePointer[DType.float32],
    w: UnsafePointer[DType.float32],
    out: UnsafePointer[DType.float32],
    rows: Int,
    eps: Float32,
    samples: UnsafePointer[DType.float64],
    reps: Int,
    warmup: Int,
):
    rmsnorm_rows[WIDTH](x, w, out, rows, eps)
    for _ in range(warmup):
        rmsnorm_rows[WIDTH](x, w, out, rows, eps)
    for repetition in range(reps):
        let started = perf_counter()
        rmsnorm_rows[WIDTH](x, w, out, rows, eps)
        samples.store(repetition, perf_counter() - started)


fn main():
    let rows = arg_int("--rows", 0)
    let cols = arg_int("--cols", 0)
    let reps = arg_int("--reps", 200)
    let warmup = arg_int("--warmup", 20)
    let eps = Float32(arg_float("--eps", 1e-6))
    let key = arg_text("--key", "")
    let x_path = arg_text("--x", "")
    let w_path = arg_text("--w", "")
    let y_path = arg_text("--y", "")
    let out_path = arg_text("--out", "")

    if rows <= 0 or cols <= 0 or x_path == "" or w_path == "":
        print('{"error": "usage: driver --key K --x FILE --w FILE [--y FILE] [--out FILE] --rows N --cols N"}')
        exit(2)

    let count = rows * cols
    let x = read_f32(x_path, count)
    let w = read_f32(w_path, cols)
    let out = UnsafePointer[DType.float32](count)
    let samples = UnsafePointer[DType.float64](reps)

    if cols == 512:
        dispatch[512](x.pointer, w.pointer, out, rows, eps, samples, reps, warmup)
    elif cols == 384:
        dispatch[384](x.pointer, w.pointer, out, rows, eps, samples, reps, warmup)
    elif cols == 1024:
        dispatch[1024](x.pointer, w.pointer, out, rows, eps, samples, reps, warmup)
    elif cols == 256:
        dispatch[256](x.pointer, w.pointer, out, rows, eps, samples, reps, warmup)
    elif cols == 64:
        dispatch[64](x.pointer, w.pointer, out, rows, eps, samples, reps, warmup)
    else:
        print('{"error": "width ' + str(cols) + ' is not compiled; add it to the dispatch list"}')
        exit(3)

    sort_samples(samples, reps)
    let ms_median = samples.load(reps // 2) * 1000.0

    var max_abs: Float64 = 0.0
    var sum_sq_delta: Float64 = 0.0
    var sum_sq_reference: Float64 = 0.0
    var checked = False
    if y_path != "":
        checked = True
        let expected = read_f32(y_path, count)
        for index in range(count):
            let delta = Float64(out.load(index)) - Float64(expected.pointer.load(index))
            let magnitude = abs(delta)
            if magnitude > max_abs:
                max_abs = magnitude
            sum_sq_delta += delta * delta
            sum_sq_reference += Float64(expected.pointer.load(index)) * Float64(expected.pointer.load(index))
    write_f32(out_path, out, count)

    var rel_l2: Float64 = 0.0
    if sum_sq_reference > 0.0:
        rel_l2 = (sum_sq_delta / sum_sq_reference) ** 0.5
    let traffic = Float64(2 * count + cols) * 4.0
    let gbps = traffic / (ms_median * 1e-3) / 1e9

    print(
        "{"
        + '"backend": "mojo_cpu",'
        + '"key": "'
        + key
        + '",'
        + '"kernel": "rmsnorm_scalar_row_per_thread",'
        + '"rows": '
        + str(rows)
        + ','
        + '"cols": '
        + str(cols)
        + ','
        + '"dtype": "fp32",'
        + '"reps": '
        + str(reps)
        + ','
        + '"warmup": '
        + str(warmup)
        + ','
        + '"ms_median": '
        + str(ms_median)
        + ','
        + '"gb_s": '
        + str(gbps)
        + ','
        + '"max_abs_err": '
        + str(max_abs)
        + ','
        + '"rel_l2_err": '
        + str(rel_l2)
        + ','
        + '"reference_checked": '
        + ("true" if checked else "false")
        + '}'
    )
