"""Minimal MINIDUMP reader: prints the exception record and the faulting module."""
import struct
import sys

STREAM_MODULE_LIST = 4
STREAM_EXCEPTION = 6
STREAM_SYSTEM_INFO = 7

EXCEPTION_NAMES = {
    0xC0000005: "ACCESS_VIOLATION",
    0xC0000006: "IN_PAGE_ERROR",
    0xC0000017: "NO_MEMORY",
    0xC000001D: "ILLEGAL_INSTRUCTION",
    0xC0000025: "NONCONTINUABLE_EXCEPTION",
    0xC0000026: "INVALID_DISPOSITION",
    0xC000008C: "ARRAY_BOUNDS_EXCEEDED",
    0xC000008E: "FLT_DIVIDE_BY_ZERO",
    0xC0000090: "FLT_INVALID_OPERATION",
    0xC0000091: "FLT_OVERFLOW",
    0xC0000093: "FLT_UNDERFLOW",
    0xC0000094: "INT_DIVIDE_BY_ZERO",
    0xC0000095: "INT_OVERFLOW",
    0xC00000FD: "STACK_OVERFLOW",
    0xC0000409: "STACK_BUFFER_OVERRUN / __fastfail",
    0xC0000374: "HEAP_CORRUPTION",
    0xE06D7363: "C++ EXCEPTION (throw)",
    0x80000003: "BREAKPOINT",
}


def read_streams(data):
    magic, version, count, rva = struct.unpack_from("<4sIII", data, 0)
    if magic != b"MDMP":
        raise SystemExit("not a minidump: {!r}".format(magic))
    streams = {}
    for i in range(count):
        stream_type, size, location = struct.unpack_from(
            "<III", data, rva + i * 12)
        streams[stream_type] = (size, location)
    return streams


def read_utf16(data, rva):
    length = struct.unpack_from("<I", data, rva)[0]
    return data[rva + 4:rva + 4 + length].decode("utf-16-le", "replace")


def modules(data, streams):
    if STREAM_MODULE_LIST not in streams:
        return []
    _, rva = streams[STREAM_MODULE_LIST]
    count = struct.unpack_from("<I", data, rva)[0]
    entries = []
    for i in range(count):
        base, size, _, _, name_rva = struct.unpack_from(
            "<QIIiI", data, rva + 4 + i * 108)
        entries.append((base, size, read_utf16(data, name_rva)))
    return entries


def main(path):
    with open(path, "rb") as handle:
        data = handle.read()
    streams = read_streams(data)
    print("== {} ==".format(path))

    if STREAM_EXCEPTION not in streams:
        print("  no exception stream (dump was taken manually?)")
        return
    _, rva = streams[STREAM_EXCEPTION]
    thread_id = struct.unpack_from("<I", data, rva)[0]
    code, flags, _record, address = struct.unpack_from("<IIQQ", data, rva + 8)
    param_count = struct.unpack_from("<I", data, rva + 32)[0]
    params = struct.unpack_from(
        "<{}Q".format(min(param_count, 15)), data, rva + 40)

    print("  thread            : {}".format(thread_id))
    print("  exception code    : 0x{:08X}  {}".format(
        code, EXCEPTION_NAMES.get(code, "unknown")))
    print("  flags             : 0x{:08X}{}".format(
        flags, "  (NONCONTINUABLE)" if flags & 1 else ""))
    print("  faulting address  : 0x{:016X}".format(address))
    if code == 0xC0000005 and len(params) >= 2:
        kind = {0: "read", 1: "write", 8: "execute"}.get(params[0], params[0])
        print("  access            : {} of 0x{:016X}".format(kind, params[1]))

    for base, size, name in modules(data, streams):
        if base <= address < base + size:
            print("  faulting module   : {} (base 0x{:X}, +0x{:X})".format(
                name, base, address - base))
            break
    else:
        if address:
            print("  faulting module   : <not in any loaded module>")


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        main(argument)
        print()
