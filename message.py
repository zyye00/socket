import pickle
import struct
from dataclasses import dataclass

from numpy.typing import NDArray


class ConnectionClosedError(EOFError):
    """Raised when peer closes the socket while reading a message."""


@dataclass
class Task:
    """Task: server → driver"""
    id: int
    coords: NDArray


@dataclass
class Result:
    """Result: driver → server"""
    id: int
    energy: float
    forces: NDArray


def send_message(conn, obj):
    data = pickle.dumps(obj)
    conn.sendall(struct.pack("I", len(data)) + data)


def recv_message(conn):
    # recv 4 bytes for length
    length_bytes = conn.recv(4)
    if len(length_bytes) == 0:
        raise ConnectionClosedError("Peer closed connection")
    if len(length_bytes) < 4:
        raise ConnectionError("Incomplete message length header")

    length = struct.unpack("I", length_bytes)[0]

    # recv the actual data
    data = b""
    while len(data) < length:
        remaining = length - len(data)
        chunk = conn.recv(min(4096, remaining))
        if not chunk:
            raise ConnectionClosedError("Peer closed connection while reading data")
        data += chunk

    return pickle.loads(data)
