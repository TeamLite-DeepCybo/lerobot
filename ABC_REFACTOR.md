# 分支说明：`refactor/abc-policy-source`

**本分支是原实现的策略源重构**：把 LeRobot 异步推理链路的"逐拍下发"接到
[`lite_abstract_policy_source`](https://github.com/TeamLite-DeepCybo/lite_abstract_policy_source)
的抽象基类上，动作下发由 `dcpolicy.ActionSource` 的子类统一实现。单独分支维护，
不随上游 LeRobot 的日常 rebase 改动（只跟踪 `main` 的安全修复）。

## 原链路与本分支的关系

LeRobot 的部署链路（未改动）：

```
PolicyServer (gRPC + pickle)  ──actions_per_chunk──►  RobotClient.action_queue
                                                          │
                                        control_loop_action()：弹一条 → robot.send_action()
```

本分支新增的抽象层链路（替换上面最后一步，服务器与传输完全不动）：

```
RobotClient.action_queue ─► QueueChunkTaker ─► LeRobotActionSource(ModelSource)
                                                  └─ ControlLoop ─► MoveControllerSink
                                                       └→ /lite_move_controller/joint_target
```

换来三件抽象层已经定义好的事：

1. **统一的动作契约**：`joint_schema_from_names(robot.action_features)` 按
   `RobotClient._action_tensor_to_action_dict` 的同一顺序建 16 维契约
   （臂 rad / 夹爪 m），顺序不对会在构造期就报错，而不是把关节发错位。
2. **超龄作废**：服务器打嗝时把整块丢掉并立刻重取，不会把几百毫秒前的动作当成当前动作。
3. **生效值可追溯**：`ControlLoop` 记录 `requested/applied`，`MoveStatus` → `SinkFeedback`，
   下游 FAULT/未授权当拍停发。

**单写者**：本链路接管 `control_loop_action()` 的职责。启用后不要再让
`RobotClient.start()` 里的控制线程同时下发，二者只能有一个写机器人。

## 新增文件

| 文件 | 作用 |
|---|---|
| `src/lerobot/abc_policy_source.py` | `joint_schema_from_names()`、`actions_to_matrix()`、`QueueChunkTaker`（从 `action_queue` 连续弹块，并保持 `latest_action` / `action_queue_size` bookkeeping）、`LeRobotActionSource(ModelSource)` |
| `tests/abc/test_abc_policy_source.py` | 19 条无 ROS / 无 gRPC 用例：`SourceConformance` 全套、契约顺序、张量/数组/列表三种输入、队列弹块与 bookkeeping、块耗尽再取、超龄作废、坏块不打断，以及经 `MoveControllerSink` 下发 `MoveTargetArray` 的端到端断言 |
| `tests/abc/conftest.py` | 局部注册 `contract/hardware/slow` 标记（不改仓库根部 pytest 配置） |

`policy_server.py` / `robot_client.py` / `transport/` 一行未改。

## 为什么保留它们的队列语义

`RobotClient` 的 `latest_action` 与 `must_go` 决定"还要不要再向服务器要新块"，被
`QueueChunkTaker` 原样维护；`chunk_size_threshold` 与 `aggregate_fn`（客户端侧的块拼接）
也继续生效——抽象层只接管"弹出来之后怎么下发"，不重写调度。

## 跑法

```bash
pip install "git+https://github.com/TeamLite-DeepCybo/lite_abstract_policy_source@main"
PYTHONPATH=src python -m pytest tests/abc/test_abc_policy_source.py -q   # 19 passed，无网络

# 接线示意（部署侧）：
client = RobotClient(cfg); client.start()            # 只做观测上行与收块
taker = QueueChunkTaker(client, max_frames=cfg.execution_horizon)
source = LeRobotActionSource(clock=MonotonicClock("control"),
                             action_names=client.robot.action_features,
                             take_chunk=taker, execute_steps=cfg.execution_horizon)
sink = MoveControllerSink(source.schema, clock=..., publish_joints=pub,
                          read_state=read_joint_states, frame_id="world_root",
                          timeout_ms=60.0, status_source=status_snapshot)
loop = ControlLoop(sink, SourceArbiter(policy=source, clock=clock), clock=clock)
# 定时器里 loop.tick()，并关闭 RobotClient 自带的 control_loop_action 线程
```

## 还没做的 / 需要确认

1. **单位（已用备注写明要求）**：契约固定 SI —— 臂 `rad`、夹爪 `m`；
   `src/lerobot/abc_policy_source.py` 的模块 docstring 与 `joint_schema_from_names()`
   都写明了这条要求，并明确**本层不做换算**。若数据集仍是"夹爪归一化开度 0..1"，
   转换请放在服务器侧或上游（与 PI05 的 `norm2si` 同理），不要藏在下发层。
2. **观测上行**：仍是 `RobotClient.control_loop_observation()`（相机 + 关节 → `TimedObservation`），
   未纳入抽象层；与"16 维 joint_states 由谁发布"这条未决项一起处理。
3. **`aggregate_fn` 的语义**：客户端会把新旧块按 timestep 聚合（默认加权平均）。
   抽象层目前只消费聚合后的结果；如果以后要在 ActionPacket 上表达聚合来源，
   可以在 `meta` 里带上 `timestep`。
4. **RTC / HIL**：抽象层已留 `RTCHook` 与仲裁器扩展点，本分支未接。
