from libc.stdio import fclose, fopen, fread, fwrite
from sys import exit

struct FloatArray:
    var pointer: UnsafePointer[DType.float32]
    var count: Int

    fn __del__(owned self: Self):
        self.pointer.free()


fn read_f32(path: String, count: Int) -> FloatArray:
    let stream = fopen(path, "rb")
    if stream.is_null():
        print("cannot open " + path)
        exit(2)
    let buffer = UnsafePointer[DType.float32](count)
    let items = fread(buffer, count * 4, 1, stream)
    fclose(stream)
    if items == 0:
        print("short read: " + path)
        exit(2)
    return FloatArray(pointer=buffer, count=count)


fn write_f32(path: String, values: UnsafePointer[DType.float32], count: Int):
    if path == "":
        return
    let stream = fopen(path, "wb")
    if stream.is_null():
        print("cannot write " + path)
        exit(2)
    fwrite(values, count * 4, 1, stream)
    fclose(stream)
