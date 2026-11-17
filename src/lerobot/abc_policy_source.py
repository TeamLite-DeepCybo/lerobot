"""把 LeRobot 异步推理链路接进策略层抽象（dcpolicy）。

LeRobot 的部署链路是 ``PolicyServer（gRPC/pickle）↔ RobotClient（本地动作队列）``：
服务器按 ``actions_per_chunk`` 返回动作块，客户端把 ``TimedAction`` 放进
``action_queue``，由 ``control_loop_action()`` 逐拍弹出并 ``robot.send_action()``。

接入抽象层后，**逐拍弹出与下发**改由抽象层负责：

    RobotClient.action_queue ─► QueueChunkTaker ─► LeRobotActionSource(ModelSource)
                                                        └─ ControlLoop ─► CommandSink

本模块只做"队列 → ActionPacket"的转换与契约声明，不重写 gRPC、不改服务器，
也不在控制线程里等网络：``take_chunk`` 必须是非阻塞的。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dcpolicy.clock import Clock
from dcpolicy.errors import ContractError
from dcpolicy.schema import DEFAULT_CONTROL_PERIOD, ActionSchema, AxisKind, Dim, Semantics
from dcpolicy.source import ModelSource

# LeRobot 的动作键形如 ``<name>.pos``；夹爪按 SI 米（与本仓库其余链路一致）。
GRIPPER_KEYWORDS = ("gripper", "jaw", "finger")


def _is_gripper(name: str) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in GRIPPER_KEYWORDS)


def joint_schema_from_names(
    names: Sequence[str],
    *,
    control_period: float = DEFAULT_CONTROL_PERIOD,
    frame: str = "joints",
) -> ActionSchema:
    """按 ``robot.action_features`` 的键顺序生成动作契约。

    LeRobot 的动作张量是按 ``robot.action_features`` 顺序排列的（见
    ``RobotClient._action_tensor_to_action_dict``），所以这里必须**照抄顺序**，
    不能重排；夹爪维度按名称识别为 SI 米。
    """
    if not names:
        raise ContractError("action_features 为空，无法生成动作契约")
    dims = tuple(
        Dim(
            name,
            kind=AxisKind.GRIPPER if _is_gripper(name) else AxisKind.POSITION,
            unit="m" if _is_gripper(name) else "rad",
        )
        for name in names
    )
    return ActionSchema(
        dims=dims,
        semantics=Semantics.ABSOLUTE,
        frame=frame,
        control_period=control_period,
        group="lerobot_action_features",
    )


def actions_to_matrix(actions: Sequence[Any], *, expected_dim: int) -> np.ndarray:
    """一撮动作（torch tensor / ndarray / list）→ ``(N, D)`` float64。"""
    if not actions:
        raise ContractError("没有动作可供组装")
    rows = []
    for action in actions:
        array = action.detach().cpu().numpy() if hasattr(action, "detach") else np.asarray(action)
        row = np.asarray(array, dtype=np.float64).reshape(-1)
        if row.shape[0] != expected_dim:
            raise ContractError(f"动作维度 {row.shape[0]} 与契约 {expected_dim} 不符")
        rows.append(row)
    matrix = np.stack(rows)
    if not np.all(np.isfinite(matrix)):
        raise ContractError("动作含 NaN/Inf")
    return matrix


class QueueChunkTaker:
    """从 ``RobotClient.action_queue`` 连续弹出最多 ``max_frames`` 条动作。

    保持原客户端的 bookkeeping（``latest_action`` / ``action_queue_size``）：
    这两项决定了客户端是否继续向服务器请求新块，不能因为换了下发路径就断掉。
    """

    def __init__(self, client: Any, *, max_frames: int = 8) -> None:
        if max_frames <= 0:
            raise ValueError("max_frames 必须为正")
        self._client = client
        self._max_frames = int(max_frames)
        self.taken = 0
        self.empty = 0

    def __call__(self) -> Any | None:
        client = self._client
        actions: list[Any] = []
        latest_timestep = None
        with client.action_queue_lock:
            client.action_queue_size.append(client.action_queue.qsize())
            while len(actions) < self._max_frames:
                try:
                    timed_action = client.action_queue.get_nowait()
                except Exception:  # queue.Empty（不 import 具体异常类型）
                    break
                actions.append(timed_action.get_action())
                latest_timestep = timed_action.get_timestep()
        if not actions:
            self.empty += 1
            return None
        if latest_timestep is not None:
            with client.latest_action_lock:
                client.latest_action = latest_timestep
        self.taken += len(actions)
        return actions

    def stats(self) -> Mapping[str, int]:
        return {"taken": self.taken, "empty": self.empty}


class LeRobotActionSource(ModelSource):
    """LeRobot 动作队列 → ``ActionPacket``（逐帧 ``execute_steps`` 由配置决定）。

    ``take_chunk`` 返回一组动作（列表 / 二维数组）或 ``None``；转换与维度校验在这里。
    """

    def __init__(
        self,
        *,
        clock: Clock,
        action_names: Sequence[str],
        take_chunk: Callable[[], Any | None],
        schema: ActionSchema | None = None,
        execute_steps: int | None = None,
        post: Any | None = None,
        max_backlog_frames: int = 5,
        max_obs_age_frames: int = 5,
        control_period: float = DEFAULT_CONTROL_PERIOD,
        version: str = "",
    ) -> None:
        self._schema = schema or joint_schema_from_names(
            action_names, control_period=control_period
        )
        super().__init__(
            None,  # 远端喂块：本地不跑模型
            clock=clock,
            schema=self._schema,
            post=post,
            execute_steps=execute_steps,
            max_backlog_frames=max_backlog_frames,
            max_obs_age_frames=max_obs_age_frames,
        )
        self._take_chunk = take_chunk
        self._dt = float(self._schema.control_period)
        self._version = version
        self._chunks_taken = 0
        self._bad_chunks = 0
        self.name = "lerobot-async" + (f":{version}" if version else "")

    def poll(self, obs, now: float):
        packet = super().poll(obs, now)
        if packet is not None:
            return packet
        if self.stats()["queued"] == 0:
            raw = self._take_chunk()
            if raw is None:
                return None
            try:
                matrix = actions_to_matrix(list(raw), expected_dim=self._schema.ndim)
            except ContractError:
                self._bad_chunks += 1
                return None
            self.submit_chunk(self._to_action_chunk(matrix, now))
            return super().poll(obs, now)
        return None

    def stats(self) -> Mapping[str, object]:
        base = dict(super().stats())
        base.update({"chunks_taken": self._chunks_taken, "bad_chunks": self._bad_chunks})
        return base

    def _to_action_chunk(self, matrix: np.ndarray, now: float):
        from dcpolicy.types import ActionChunk, SourceRef

        self._chunks_taken += 1
        return ActionChunk(
            values=matrix,
            schema=self._schema,
            dt=self._dt,
            t_predict=float(now),
            source=SourceRef(kind="model", name=self.name, version=self._version),
            chunk_id=self._chunks_taken,
        )
