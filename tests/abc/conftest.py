"""为 ABC 契约测试注册标记，避免污染仓库根部的 pytest 配置。"""

from __future__ import annotations


def pytest_configure(config):
    for line in (
        "contract: 抽象层契约断言（无硬件、无 GPU、无推理服务）",
        "hardware: 需要真机/相机/推理服务",
        "slow: 慢用例",
    ):
        config.addinivalue_line("markers", line)
