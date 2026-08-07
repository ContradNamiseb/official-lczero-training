"""Serve training batches from the WSL data loader over localhost.

Phase 2 of docs/directml_training_port.md. Keeps the existing C++ loader
where it works and ships completed batches to the native Windows trainer.

Nothing here imports PyTorch, and the C++ extension is imported lazily
inside :class:`DataLoaderSource`, so this module is importable (and
testable, via :class:`BatchSource`) on a machine that has neither.
"""

from __future__ import annotations

import logging
import queue
import socket
import threading
from collections.abc import Sequence
from contextlib import suppress
from typing import Protocol

import numpy as np

from . import batch_protocol as wire

logger = logging.getLogger(__name__)

# Batches held between the loader and the socket. Bounded so a trainer
# slower than the loader stalls the loader instead of growing without
# limit -- one batch of the target config is about 1.2 MB.
DEFAULT_QUEUE_DEPTH = 4

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18765

_POLL_SECONDS = 0.25


class BatchSource(Protocol):
    """Anything that can produce successive training batches."""

    def next_batch(self) -> tuple[np.ndarray, ...]: ...

    def close(self) -> None: ...


class DataLoaderSource:
    """The real C++ pipeline, started from ``config.data_loader``."""

    def __init__(self, data_loader_config) -> None:
        # Imported here, not at module scope: the extension is a Linux .so,
        # and this module must stay importable on Windows for the tests.
        from lczero_training.dataloader import make_dataloader

        self._loader = make_dataloader(data_loader_config)

    def next_batch(self) -> tuple[np.ndarray, ...]:
        return self._loader.get_next()

    def close(self) -> None:
        with suppress(Exception):
            self._loader.stop()


class BatchServer:
    """Streams batches to one client at a time.

    The loader outlives any single client: `lc0-directml-train` is meant to
    be re-run repeatedly, and rebuilding the shuffling pool for every
    invocation would dominate the run. Between clients the producer simply
    blocks on a full queue.
    """

    def __init__(
        self,
        source: BatchSource,
        config_hash: bytes,
        *,
        queue_depth: int = DEFAULT_QUEUE_DEPTH,
        exit_on_disconnect: bool = False,
    ) -> None:
        self._source = source
        self._config_hash = config_hash
        self._exit_on_disconnect = exit_on_disconnect
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, queue_depth))
        self._stop = threading.Event()
        self._producer: threading.Thread | None = None
        self._specs: tuple[wire.TensorSpec, ...] = ()
        self._sequence = 0

    @property
    def specs(self) -> tuple[wire.TensorSpec, ...]:
        return self._specs

    def stop(self) -> None:
        """Ask the server to wind down; safe to call from a signal handler."""
        self._stop.set()

    def prime(self) -> None:
        """Fetch one batch to learn the tensor specs, then start producing.

        The probe batch is kept and queued rather than discarded -- the
        first batch out of the pipeline is expensive.
        """
        logger.info("Fetching first batch to determine tensor specs")
        first = self._source.next_batch()
        self._specs = tuple(wire.TensorSpec.from_array(a) for a in first)
        for index, spec in enumerate(self._specs):
            logger.info(
                "  tensor %d: %s %s (%d bytes)",
                index,
                spec.dtype,
                spec.shape,
                spec.nbytes,
            )
        self._queue.put(first)
        self._producer = threading.Thread(
            target=self._produce, name="batch-producer", daemon=True
        )
        self._producer.start()

    def _produce(self) -> None:
        while not self._stop.is_set():
            try:
                batch = self._source.next_batch()
            except Exception:
                logger.exception("Data loader failed; stopping")
                self._stop.set()
                return
            # Bounded put with a poll, so a stalled client cannot keep this
            # thread blocked past a shutdown request.
            while not self._stop.is_set():
                try:
                    self._queue.put(batch, timeout=_POLL_SECONDS)
                    break
                except queue.Full:
                    continue

    def serve_forever(
        self,
        host: str,
        port: int,
        ready: threading.Event | None = None,
    ) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.settimeout(_POLL_SECONDS)
        try:
            listener.bind((host, port))
            listener.listen(1)
            self.port = listener.getsockname()[1]
            logger.info("Listening on %s:%d", host, self.port)
            # Set only once the socket is accepting, so a caller waiting on
            # this cannot race the bind.
            if ready is not None:
                ready.set()
            while not self._stop.is_set():
                try:
                    connection, peer = listener.accept()
                except TimeoutError:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                logger.info("Client connected from %s:%d", *peer)
                try:
                    self._serve_client(connection)
                except (ConnectionError, OSError) as error:
                    logger.info("Client disconnected: %s", error)
                finally:
                    with suppress(OSError):
                        connection.shutdown(socket.SHUT_RDWR)
                    connection.close()
                    logger.info("Client connection closed")
                if self._exit_on_disconnect:
                    break
        finally:
            listener.close()
            self.shutdown()

    def _serve_client(self, connection: socket.socket) -> None:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        handshake = wire.Handshake(
            protocol_version=wire.PROTOCOL_VERSION,
            config_hash=self._config_hash,
            tensors=self._specs,
        )
        connection.sendall(handshake.to_bytes())

        read = wire.make_reader(connection)
        verdict = read(1)[0]
        if verdict != wire.VERDICT_ACCEPT:
            reason = wire.read_error_message(read)
            logger.error("Client rejected the stream: %s", reason)
            return
        logger.info("Client accepted the stream; streaming batches")

        while not self._stop.is_set():
            try:
                batch = self._queue.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                continue
            try:
                parts = wire.encode_batch(self._sequence, batch, self._specs)
            except wire.ProtocolError as error:
                logger.error("%s", error)
                with suppress(OSError):
                    connection.sendall(wire.encode_error(str(error)))
                self._stop.set()
                return
            for part in parts:
                connection.sendall(part)
            self._sequence += 1

        with suppress(OSError):
            connection.sendall(wire.encode_end())

    def shutdown(self) -> None:
        """Stop the producer and the loader. Idempotent."""
        self._stop.set()
        if self._producer is not None and self._producer.is_alive():
            # Drain so a producer blocked on a full queue can notice the
            # stop event and exit instead of being killed mid-batch.
            with suppress(queue.Empty):
                while True:
                    self._queue.get_nowait()
            self._producer.join(timeout=5.0)
            if self._producer.is_alive():
                logger.warning("Producer thread did not stop in time")
        self._producer = None
        logger.info("Stopping data loader")
        self._source.close()


def serve(
    source: BatchSource,
    config_hash: bytes,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    queue_depth: int = DEFAULT_QUEUE_DEPTH,
    exit_on_disconnect: bool = False,
    ready: threading.Event | None = None,
) -> BatchServer:
    """Prime and run a server until it is stopped. Returns it for shutdown.

    ``ready`` is set once the listening socket is accepting, which the
    tests use to avoid racing the accept loop. Pass ``port=0`` to let the
    OS choose one and read it back from ``server.port``.
    """
    server = BatchServer(
        source,
        config_hash,
        queue_depth=queue_depth,
        exit_on_disconnect=exit_on_disconnect,
    )
    server.prime()
    server.serve_forever(host, port, ready=ready)
    return server


def specs_for(arrays: Sequence[np.ndarray]) -> tuple[wire.TensorSpec, ...]:
    return tuple(wire.TensorSpec.from_array(array) for array in arrays)
