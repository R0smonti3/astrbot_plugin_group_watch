from __future__ import annotations

import asyncio
import os
import re
import tempfile
from typing import Any

from astrbot.api import logger
from astrbot.core.message.components import Image

try:
    from .constants import IMAGE_WAIT_SECONDS, MAX_IMAGES_PER_MESSAGE
except (ImportError, ValueError):
    from constants import IMAGE_WAIT_SECONDS, MAX_IMAGES_PER_MESSAGE


class ImageService:
    """管理群聊图片的异步转述、摘要生成与临时文件清理。"""

    def __init__(self) -> None:
        self._image_tasks: set[asyncio.Task] = set()

    @property
    def image_tasks(self) -> set[asyncio.Task]:
        return self._image_tasks

    @staticmethod
    def extract_prompt_images(req: Any) -> list[str]:
        """从 AstrBot 请求对象提取图片文件路径或 URL。"""
        images: list[str] = []
        for prompt_item in getattr(req, "prompt", []) or []:
            if isinstance(prompt_item, Image):
                p = getattr(prompt_item, "path", None) or getattr(prompt_item, "url", None)
                if p:
                    images.append(str(p))
            elif isinstance(prompt_item, dict) and prompt_item.get("type") == "image":
                url = prompt_item.get("url") or prompt_item.get("image")
                if url:
                    images.append(str(url))
        return images[:MAX_IMAGES_PER_MESSAGE]

    def create_describe_task(
        self,
        provider: Any,
        session_id: str,
        entry: dict[str, Any],
        images: list[str],
        char_limit: int = 200,
    ) -> asyncio.Task:
        entry["image_state"] = "pending"
        task = asyncio.create_task(
            self.describe_images(provider, session_id, entry, images, char_limit)
        )
        entry["_image_task"] = task
        self._image_tasks.add(task)
        task.add_done_callback(self._image_tasks.discard)
        return task

    async def describe_images(
        self,
        provider: Any,
        session_id: str,
        entry: dict[str, Any],
        images: list[str],
        char_limit: int = 200,
    ) -> None:
        """异步调用多模态模型生成图片摘要。"""
        try:
            if provider is None:
                entry["image_state"] = "failed"
                return
            if not getattr(provider, "multimodal", False):
                entry["image_state"] = "unsupported"
                return

            limit = max(40, char_limit)
            prompt = (
                f"用不超过{limit}字简述这张/这几张图片的内容（是什么、主体、文字、梗意），"
                f"只输出一段纯文本，不要任何前缀、废话或 Markdown。"
            )
            response = await asyncio.wait_for(
                provider.text_chat(
                    prompt=prompt,
                    image_urls=images,
                    system_prompt="",
                    contexts=[],
                    session_id=session_id,
                ),
                timeout=25,
            )
            text = str(getattr(response, "completion_text", "") or "").strip()
            if text:
                text = re.sub(r"\s+", " ", text)[:limit]
                entry["image_desc"] = text
                entry["image_state"] = "ready"
                logger.info(f"[group_watch] 图片摘要已生成 ({session_id}): {text[:30]}...")
            else:
                entry["image_state"] = "empty"
        except Exception as exc:  # noqa: BLE001
            entry["image_state"] = "error"
            logger.debug(f"[group_watch] 图片摘要生成失败: {exc}")
        finally:
            entry.pop("_image_task", None)
            for path in images:
                if not isinstance(path, str) or not path.startswith(tempfile.gettempdir()):
                    continue
                try:
                    os.remove(path)
                except OSError:
                    pass

    @staticmethod
    async def wait_images(
        session_id: str,
        count: int,
        buffers: dict[str, list[dict[str, Any]]],
        wait_seconds: float = IMAGE_WAIT_SECONDS,
    ) -> None:
        """在进行接话评估前，给后台仍在生成的图片摘要预留适当等待时间。"""
        tasks = []
        for item in (buffers.get(session_id) or [])[-max(1, count):]:
            task = item.get("_image_task")
            if (
                item.get("image_state") == "pending"
                and isinstance(task, asyncio.Task)
                and not task.done()
            ):
                tasks.append(task)
        if tasks:
            await asyncio.wait(tasks, timeout=wait_seconds)

    def cancel_all(self) -> None:
        for task in list(self._image_tasks):
            if not task.done():
                task.cancel()
        self._image_tasks.clear()
