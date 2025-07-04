import os
import json
import requests
import atexit
import sys
from datetime import datetime
# ---------- Rich Console Logger 替代 ----------
from types import SimpleNamespace
try:
    from rich.console import Console
    console: Console | None = Console()
except ImportError:
    console = None  # type: ignore

# 提供与 loguru 类似接口
if console:
    logger = SimpleNamespace(
        info=lambda message, *args, **kwargs: console.log(message, *args, **kwargs),  # type: ignore[union-attr]
        warning=lambda message, *args, **kwargs: console.log(f"[bold yellow]WARNING[/] {message}", *args, **kwargs),  # type: ignore[union-attr]
        error=lambda message, *args, **kwargs: console.log(f"[bold red]ERROR[/] {message}", *args, **kwargs),  # type: ignore[union-attr]
    )
else:
    logger = SimpleNamespace(
        info=lambda message, *args, **kwargs: print(message),
        warning=lambda message, *args, **kwargs: print(message),
        error=lambda message, *args, **kwargs: print(message),
    )

from typing import Optional

from src.util.llm_client.base_llm_client import BaseLLMClient


class APIModelClient(BaseLLMClient):
    def __init__(
            self,
            config: dict,
            prompt_dir: Optional[str]=None,
            output_dir: Optional[str]=None,
        ):
        super().__init__(config, prompt_dir, output_dir)

        # 初始化成本统计变量
        self.request_count: int = 0  # 已发送请求数量
        self.total_tokens: int = 0   # 已消耗 token 数量

        # 在程序退出前输出统计信息
        atexit.register(self._print_cost_stats)

        self.url = config["url"]
        model = config["model"]
        stream = config.get("stream", False)
        top_p = config.get("top-p", 0.7)
        temperature = config.get("temperature", 0.95)
        max_tokens = config.get("max_tokens", 3200)
        seed = config.get("seed", None)
        api_key = config["api_key"]
        self.max_attempts = config.get("max_attempts", 50)
        self.sleep_time = config.get("sleep_time", 60)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.payload = {
            "model": model,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed
        }

        # 无需额外日志配置，已切换到 Rich Console

    def reset(self, output_dir: Optional[str]=None) -> None:
        self.messages = []
        if output_dir is not None:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

    def chat_once(self) -> str:
        """发送一次聊天请求，并更新成本统计信息"""
        self.payload["messages"] = self.messages
        response = requests.request("POST", self.url, json=self.payload, headers=self.headers)

        # 解析响应
        resp_json = json.loads(response.text)

        # 更新请求计数
        self.request_count += 1

        # 如果返回中包含 usage 字段，则累加 token 数量
        usage_info = resp_json.get("usage", {})
        current_tokens = usage_info.get("total_tokens", 0)
        self.total_tokens += current_tokens

        # 输出日志
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"本次 tokens: {current_tokens}, 已请求次数: {self.request_count}, 累计 tokens: {self.total_tokens}")

        response_content = resp_json["choices"][-1]["message"]["content"]
        return response_content

    def _print_cost_stats(self) -> None:
        """程序退出时输出请求次数和 token 使用量"""
        logger.info(f"[APIModelClient] 总请求次数: {self.request_count}, 总 tokens 消耗: {self.total_tokens}")
