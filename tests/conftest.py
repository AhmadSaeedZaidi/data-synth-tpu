"""Suite-wide safety guard: tests must never open real network sockets."""

from __future__ import annotations

import socket

import pytest


@pytest.fixture(autouse=True)
def forbid_real_socket_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any test attempts a real socket connection."""

    def denied_connect(sock: socket.socket, address) -> None:
        raise AssertionError(f"real network connection attempted: {address!r}")

    def denied_connect_ex(sock: socket.socket, address) -> int:
        raise AssertionError(f"real network connection attempted: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", denied_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", denied_connect_ex)
