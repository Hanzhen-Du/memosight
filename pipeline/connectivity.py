"""Connectivity: a switchable is_online() mock.

On a real device this would ping the gateway or the cloud. In this phase it is a manually
switchable mock, so tests can walk the online, offline and recovery paths deterministically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Connectivity(ABC):
    """Abstract connectivity probe."""

    @abstractmethod
    def is_online(self) -> bool:
        raise NotImplementedError


class ConnectivityMock(Connectivity):
    """A connectivity mock that can be switched by hand."""

    def __init__(self, online: bool = True):
        self._online = online

    def is_online(self) -> bool:
        return self._online

    def set_online(self, value: bool) -> None:
        self._online = bool(value)

    def go_offline(self) -> None:
        self._online = False

    def go_online(self) -> None:
        self._online = True
