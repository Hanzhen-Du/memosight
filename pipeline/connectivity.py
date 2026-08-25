"""联网状态：可切换的 is_online() mock。

真实设备上这里会去 ping 网关/云端；本阶段用一个可手动切换的 mock，
让测试能确定性地走"联网 / 断网 / 恢复"三条路径。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Connectivity(ABC):
    """联网探测抽象。"""

    @abstractmethod
    def is_online(self) -> bool:
        raise NotImplementedError


class ConnectivityMock(Connectivity):
    """可手动切换的联网 mock。"""

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
