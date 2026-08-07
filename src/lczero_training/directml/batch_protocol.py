"""Wire format for streaming training batches from WSL to Windows.

Phase 2 of docs/directml_training_port.md. The C++ data loader depends on
``inotify``, ``epoll``, ``unistd``, and ``pread``, so it stays in WSL and
completed batches are shipped over localhost instead of porting those
internals.

This module is deliberately the only piece both ends share, and it imports
nothing beyond the standard library and numpy: the server runs under Linux
with no PyTorch, and the client runs under Windows with no C++ extension.

Framing
-------

On connect the server sends one :class:`Handshake`, the client validates it
and replies with a single :class:`Verdict` byte, and the server then streams
:data:`FRAME_BATCH` frames until either side stops.

Nothing is pickled. Array payloads are the raw contiguous little-endian
bytes described by the handshake's tensor specs, in order, with no
per-frame shape metadata -- batch shapes are fixed for the life of a
connection, and the server re-checks every batch against the specs it
announced.
"""

from __future__ import annotations

import dataclasses
import hashlib
import socket
import struct
from collections.abc import Sequence

import numpy as np

MAGIC = b"LC0DMLB\x00"

# Bump on any incompatible change to the framing below. The client refuses
# to train against a server announcing a different version.
PROTOCOL_VERSION = 1

CONFIG_HASH_SIZE = 32  # sha256
_DTYPE_FIELD_SIZE = 8

FRAME_BATCH = 1
FRAME_END = 2
FRAME_ERROR = 3

VERDICT_ACCEPT = 1
VERDICT_REJECT = 0

_HANDSHAKE_HEAD = struct.Struct(f"<{len(MAGIC)}sI{CONFIG_HASH_SIZE}sI")
_SPEC_HEAD = struct.Struct(f"<{_DTYPE_FIELD_SIZE}sI")
_UINT32 = struct.Struct("<I")
_UINT64 = struct.Struct("<Q")
_FRAME_HEAD = struct.Struct("<BQ")
_ERROR_HEAD = struct.Struct("<I")


class ProtocolError(RuntimeError):
    """The peer sent something this version cannot interpret."""


class ConfigMismatch(RuntimeError):
    """Server and client were started from different configurations."""


def config_hash(data_loader_config) -> bytes:
    """Stable hash of the data loader configuration.

    Hashes ``config.data_loader`` rather than the whole root config: it is
    exactly the part that determines what batches come out, so two peers
    agreeing on it agree on the data contract. Model and optimizer settings
    differing between the two ends is legitimate and must not be rejected
    here.
    """
    serialized = data_loader_config.SerializeToString(deterministic=True)
    return hashlib.sha256(serialized).digest()


def _normalize_dtype(dtype: np.dtype) -> np.dtype:
    """Force a little-endian byte order for the wire."""
    if dtype.byteorder == ">":
        return dtype.newbyteorder("<")
    # '=' (native) and '|' (not applicable) are little-endian on every
    # platform this port targets; name them explicitly on the wire anyway.
    return np.dtype(dtype.str.replace("=", "<").replace("|", "<", 1))


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    """dtype, rank, shape, and byte length of one array in a batch."""

    dtype: str
    shape: tuple[int, ...]

    @classmethod
    def from_array(cls, array: np.ndarray) -> TensorSpec:
        return cls(
            dtype=_normalize_dtype(array.dtype).str,
            shape=tuple(int(size) for size in array.shape),
        )

    @property
    def nbytes(self) -> int:
        count = 1
        for size in self.shape:
            count *= size
        return count * np.dtype(self.dtype).itemsize

    def matches(self, array: np.ndarray) -> bool:
        return (
            _normalize_dtype(array.dtype).str == self.dtype
            and tuple(array.shape) == self.shape
        )

    def empty(self) -> np.ndarray:
        return np.empty(self.shape, dtype=np.dtype(self.dtype))

    def to_bytes(self) -> bytes:
        encoded = self.dtype.encode("ascii")
        if len(encoded) > _DTYPE_FIELD_SIZE:
            raise ProtocolError(f"dtype string too long: {self.dtype!r}")
        payload = [
            _SPEC_HEAD.pack(
                encoded.ljust(_DTYPE_FIELD_SIZE, b"\x00"), len(self.shape)
            )
        ]
        payload.extend(_UINT32.pack(size) for size in self.shape)
        payload.append(_UINT64.pack(self.nbytes))
        return b"".join(payload)

    @classmethod
    def from_reader(cls, read) -> TensorSpec:
        raw_dtype, rank = _SPEC_HEAD.unpack(read(_SPEC_HEAD.size))
        dtype = raw_dtype.rstrip(b"\x00").decode("ascii")
        try:
            np.dtype(dtype)
        except TypeError as error:
            raise ProtocolError(f"unknown dtype {dtype!r}") from error
        shape = tuple(
            _UINT32.unpack(read(_UINT32.size))[0] for _ in range(rank)
        )
        (declared,) = _UINT64.unpack(read(_UINT64.size))
        spec = cls(dtype=dtype, shape=shape)
        if declared != spec.nbytes:
            raise ProtocolError(
                f"declared byte length {declared} disagrees with "
                f"{spec.shape} of {spec.dtype} ({spec.nbytes})"
            )
        return spec


@dataclasses.dataclass(frozen=True)
class Handshake:
    """Everything the client needs to validate and preallocate."""

    protocol_version: int
    config_hash: bytes
    tensors: tuple[TensorSpec, ...]

    def to_bytes(self) -> bytes:
        if len(self.config_hash) != CONFIG_HASH_SIZE:
            raise ProtocolError("config hash must be 32 bytes")
        parts = [
            _HANDSHAKE_HEAD.pack(
                MAGIC,
                self.protocol_version,
                self.config_hash,
                len(self.tensors),
            )
        ]
        parts.extend(spec.to_bytes() for spec in self.tensors)
        return b"".join(parts)

    @classmethod
    def from_reader(cls, read) -> Handshake:
        magic, version, hash_bytes, count = _HANDSHAKE_HEAD.unpack(
            read(_HANDSHAKE_HEAD.size)
        )
        if magic != MAGIC:
            raise ProtocolError(f"not an lc0 batch stream (magic {magic!r})")
        tensors = tuple(TensorSpec.from_reader(read) for _ in range(count))
        return cls(
            protocol_version=version,
            config_hash=hash_bytes,
            tensors=tensors,
        )

    def check_compatible(self, expected_config_hash: bytes) -> None:
        """Raise unless this stream is safe to train against.

        Called by the client before the first training step, so a
        misconfigured pair fails immediately instead of silently training on
        the wrong data.
        """
        if self.protocol_version != PROTOCOL_VERSION:
            raise ProtocolError(
                f"server speaks protocol version {self.protocol_version}, "
                f"this client speaks {PROTOCOL_VERSION}"
            )
        if self.config_hash != expected_config_hash:
            raise ConfigMismatch(
                "server and client were started from different data loader "
                f"configurations (server {self.config_hash.hex()[:16]}, "
                f"client {expected_config_hash.hex()[:16]})"
            )


def encode_batch(
    sequence: int, arrays: Sequence[np.ndarray], specs: Sequence[TensorSpec]
) -> list[bytes]:
    """Frame header plus one contiguous little-endian block per array."""
    if len(arrays) != len(specs):
        raise ProtocolError(f"expected {len(specs)} arrays, got {len(arrays)}")
    parts = [_FRAME_HEAD.pack(FRAME_BATCH, sequence)]
    for array, spec in zip(arrays, specs):
        if not spec.matches(array):
            raise ProtocolError(
                f"batch tensor changed shape or dtype: announced "
                f"{spec.shape} of {spec.dtype}, got {tuple(array.shape)} of "
                f"{array.dtype.str}"
            )
        parts.append(
            np.ascontiguousarray(array, dtype=np.dtype(spec.dtype)).tobytes()
        )
    return parts


def encode_end() -> bytes:
    return _FRAME_HEAD.pack(FRAME_END, 0)


def encode_error(message: str) -> bytes:
    payload = message.encode("utf-8")
    return (
        _FRAME_HEAD.pack(FRAME_ERROR, 0)
        + _ERROR_HEAD.pack(len(payload))
        + payload
    )


def read_frame_header(read) -> tuple[int, int]:
    frame_type, sequence = _FRAME_HEAD.unpack(read(_FRAME_HEAD.size))
    return frame_type, sequence


def read_error_message(read) -> str:
    (length,) = _ERROR_HEAD.unpack(read(_ERROR_HEAD.size))
    return read(length).decode("utf-8", errors="replace")


def make_reader(connection: socket.socket):
    """An exact-length reader over a socket.

    ``recv`` is free to return short, which silently corrupts a binary
    framing if not handled; every read in this module goes through here.
    """

    def read(size: int) -> bytes:
        if size == 0:
            return b""
        chunks = []
        remaining = size
        while remaining:
            chunk = connection.recv(remaining)
            if not chunk:
                raise ConnectionError(
                    f"peer closed after {size - remaining} of {size} bytes"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks) if len(chunks) > 1 else chunks[0]

    return read


def read_into(connection: socket.socket, array: np.ndarray) -> None:
    """Fill ``array`` directly from the socket, with no intermediate copy."""
    view = memoryview(array.reshape(-1).view(np.uint8))
    offset = 0
    total = array.nbytes
    while offset < total:
        received = connection.recv_into(view[offset:], total - offset)
        if not received:
            raise ConnectionError(
                f"peer closed after {offset} of {total} payload bytes"
            )
        offset += received
