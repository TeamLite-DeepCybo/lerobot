"""LeRobot 异步推理链路的策略源契约测试：无 ROS、无 gRPC、无服务器。

覆盖：抽象层契约（SourceConformance）、按 ``action_features`` 顺序建契约、
队列弹块与 bookkeeping、块耗尽/超龄/坏块、以及经 MoveControllerSink 下发
``MoveTargetArray`` 的端到端断言。
"""

from __future__ import annotations

import threading
from queue import Empty, Queue
from types import SimpleNamespace

import numpy as np
import pytest

from dcpolicy.clock import FakeClock
from dcpolicy.conformance import SourceConformance
from dcpolicy.runtime import ControlLoop, SourceArbiter
from integrations.lite_move_controller import TARGET_JOINTS, MoveControllerSink

from lerobot.abc_policy_source import (
    LeRobotActionSource,
    QueueChunkTaker,
    actions_to_matrix,
    joint_schema_from_names,
)

pytestmark = pytest.mark.contract

ACTION_NAMES = (
    "left_shoulder_pitch.pos",
    "left_shoulder_roll.pos",
    "left_shoulder_yaw.pos",
    "left_elbow_pitch.pos",
    "left_wrist_yaw.pos",
    "left_wrist_roll.pos",
    "left_wrist_pitch.pos",
    "right_shoulder_pitch.pos",
    "right_shoulder_roll.pos",
    "right_shoulder_yaw.pos",
    "right_elbow_pitch.pos",
    "right_wrist_yaw.pos",
    "right_wrist_roll.pos",
    "right_wrist_pitch.pos",
    "left_gripper.pos",
    "right_gripper.pos",
)
DIM = len(ACTION_NAMES)


def vector(value: float = 0.1, *, gripper: float = 0.02) -> np.ndarray:
    row = np.full(DIM, value, dtype=np.float32)
    row[14] = gripper
    row[15] = gripper
    return row


class FakeTimedAction:
    def __init__(self, action, timestep: int) -> None:
        self.action = np.asarray(action, dtype=np.float32)
        self.timestep = timestep

    def get_action(self):
        return self.action

    def get_timestep(self) -> int:
        return self.timestep


class FakeClient:
    """仿 RobotClient 的队列与 bookkeeping 字段。"""

    def __init__(self, actions=()) -> None:
        self.action_queue: Queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.latest_action_lock = threading.Lock()
        self.action_queue_size: list[int] = []
        self.latest_action = -1
        for timestep, action in enumerate(actions):
            self.action_queue.put(FakeTimedAction(action, timestep))

    def push(self, action, timestep: int) -> None:
        self.action_queue.put(FakeTimedAction(action, timestep))


class ScriptedTake:
    def __init__(self, chunks) -> None:
        self.chunks = list(chunks)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.chunks.pop(0) if self.chunks else None


def make_source(take=None, **kwargs):
    clock = FakeClock(domain="control")
    take = take or ScriptedTake([[vector()]])
    source = LeRobotActionSource(
        clock=clock, action_names=ACTION_NAMES, take_chunk=take, **kwargs
    )
    return source, clock, take


class TestLeRobotActionSourceContract(SourceConformance):
    def make_source(self):
        clock = FakeClock(domain="control")
        source = LeRobotActionSource(
            clock=clock, action_names=ACTION_NAMES, take_chunk=lambda: [vector()]
        )
        return source, clock


# --- 契约 -------------------------------------------------------------------
def test_schema_follows_action_features_order() -> None:
    schema = joint_schema_from_names(ACTION_NAMES)
    assert schema.ndim == DIM
    assert schema.names == ACTION_NAMES, "动作张量顺序必须与 robot.action_features 一致"
    assert schema.dim("left_elbow_pitch.pos").unit == "rad"
    assert schema.dim("left_gripper.pos").unit == "m", "夹爪按 SI 米"


def test_non_gripper_names_default_to_radians() -> None:
    schema = joint_schema_from_names(("a.pos", "b.pos"))
    assert all(d.unit == "rad" for d in schema.dims)


def test_actions_to_matrix_accepts_tensors_lists_and_arrays() -> None:
    class FakeTensor:
        def __init__(self, value):
            self._value = np.asarray(value, dtype=np.float32)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._value

    matrix = actions_to_matrix([FakeTensor(vector()), list(vector()), vector()], expected_dim=DIM)
    assert matrix.shape == (3, DIM)
    assert matrix.dtype == np.float64


def test_dimension_mismatch_is_a_contract_error() -> None:
    from dcpolicy.errors import ContractError

    with pytest.raises(ContractError, match="动作维度"):
        actions_to_matrix([np.zeros(8)], expected_dim=DIM)


# --- 队列行为 ---------------------------------------------------------------
def test_taker_keeps_client_bookkeeping() -> None:
    client = FakeClient([vector(0.1), vector(0.2)])
    taker = QueueChunkTaker(client, max_frames=8)
    assert taker() is not None
    assert client.latest_action == 1, "必须更新 latest_action，否则客户端不再请求新块"
    assert client.action_queue_size, "必须记录队列长度统计"
    assert taker() is None and taker.empty == 1


def test_taker_respects_max_frames() -> None:
    client = FakeClient([vector(0.1) for _ in range(5)])
    taker = QueueChunkTaker(client, max_frames=2)
    batch = taker()
    assert batch is not None and len(batch) == 2
    assert client.action_queue.qsize() == 3


def test_source_consumes_chunk_then_refills() -> None:
    source, clock, take = make_source(ScriptedTake([[vector(), vector()], [vector(0.5)]]))
    assert source.poll(None, clock.now()) is not None
    assert source.poll(None, clock.now()) is not None
    assert take.calls == 1, "一块用完之前不应重复取块"
    assert source.poll(None, clock.now()) is not None
    assert take.calls == 2


def test_empty_queue_yields_no_action() -> None:
    source, clock, _ = make_source(ScriptedTake([]))
    assert source.poll(None, clock.now()) is None


def test_backlog_is_dropped_and_refilled() -> None:
    source, clock, take = make_source(
        ScriptedTake([[vector() for _ in range(4)], [vector()]]), max_backlog_frames=2
    )
    assert source.poll(None, clock.now()) is not None
    clock.advance(1.0)
    packet = source.poll(None, clock.now())
    assert packet is not None
    assert source.stats()["backlog_drops"] == 1
    assert take.calls == 2


def test_bad_chunk_is_counted_and_does_not_break_the_loop() -> None:
    source, clock, _ = make_source(ScriptedTake([[np.zeros(8)], [vector()]]))
    assert source.poll(None, clock.now()) is None
    assert source.stats()["bad_chunks"] == 1
    assert source.poll(None, clock.now()) is not None


def test_execute_steps_limits_consumed_frames() -> None:
    source, clock, _ = make_source(ScriptedTake([[vector() for _ in range(8)]]), execute_steps=2)
    assert source.poll(None, clock.now()) is not None
    assert source.poll(None, clock.now()) is not None
    assert source.stats()["queued"] == 0


# --- 端到端：队列 → 源 → 控制环 → 运控 sink --------------------------------
def test_end_to_end_from_client_queue_to_joint_targets() -> None:
    client = FakeClient([vector(0.1, gripper=0.02), vector(0.2, gripper=0.02)])
    clock = FakeClock(domain="control")
    taker = QueueChunkTaker(client, max_frames=8)
    source = LeRobotActionSource(
        clock=clock, action_names=ACTION_NAMES, take_chunk=taker, execute_steps=2
    )
    published: list[dict] = []
    sink = MoveControllerSink(
        source.schema,
        clock=clock,
        publish_joints=published.append,
        read_state=lambda: None,
        frame_id="world_root",
        timeout_ms=60.0,
    )
    loop = ControlLoop(sink, SourceArbiter(policy=source, clock=clock), clock=clock)
    results = loop.run(2, advance=1 / 30)

    assert all(r.applied is not None for r in results)
    targets = published[0]["targets"]
    assert [t["arm"] for t in targets] == ["left", "right"]
    assert all(t["target_type"] == TARGET_JOINTS for t in targets)
    assert len(targets[0]["joint_positions"]) == 7
    assert targets[0]["gripper"] == pytest.approx(0.02)
    assert targets[0]["frame_id"] == "world_root"


def test_end_to_end_holds_when_server_stops_sending() -> None:
    client = FakeClient([vector()])
    clock = FakeClock(domain="control")
    source = LeRobotActionSource(
        clock=clock, action_names=ACTION_NAMES, take_chunk=QueueChunkTaker(client, max_frames=1),
        execute_steps=1,
    )
    published: list[dict] = []
    sink = MoveControllerSink(
        source.schema, clock=clock, publish_joints=published.append,
        read_state=lambda: None, frame_id="world_root", timeout_ms=60.0,
    )
    loop = ControlLoop(sink, SourceArbiter(policy=source, clock=clock), clock=clock)
    assert loop.tick().applied is not None
    second = loop.tick()
    assert second.applied is None and len(published) == 1
