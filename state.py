from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from astrbot.api import logger

try:
    from .constants import BUFFER_LIMIT, MAX_DECISIONS
except (ImportError, ValueError):
    from constants import BUFFER_LIMIT, MAX_DECISIONS


class SilenceStorage:
    """轻量群聊静默熔断记录持久化，保证重启或热重载后静默期依然有效。"""

    def __init__(self, target_path: Path | None = None) -> None:
        if target_path is not None:
            self._path = target_path
        else:
            candidates: list[Path] = []
            try:
                from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path
                candidates.append(Path(get_astrbot_plugin_data_path()) / "astrbot_plugin_group_watch" / "silence_records.json")
            except Exception:
                pass
            candidates.append(Path.cwd() / "data" / "plugin_data" / "astrbot_plugin_group_watch" / "silence_records.json")
            candidates.append(Path(__file__).resolve().parent / "data" / "silence_records.json")

            selected_path = candidates[-1]
            for cand in candidates:
                try:
                    cand.parent.mkdir(parents=True, exist_ok=True)
                    selected_path = cand
                    break
                except (PermissionError, OSError):
                    continue
            self._path = selected_path

        self._records: dict[str, float] = {}
        self.load()

    def load(self) -> dict[str, float]:
        """从文件加载静默时间戳，并主动剔除已过期的记录。"""
        if not self._path.is_file():
            self._records = {}
            return self._records

        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                self._records = {}
                return self._records

            now = time.time()
            active: dict[str, float] = {}
            has_expired = False
            for sid, until in data.items():
                try:
                    ts = float(until)
                    if ts > now:
                        active[str(sid)] = ts
                    else:
                        has_expired = True
                except (ValueError, TypeError):
                    has_expired = True

            self._records = active
            if has_expired:
                self._save()
            return self._records
        except Exception as exc:
            logger.warning(f"[group_watch] 读取静默记录文件失败: {exc}")
            self._records = {}
            return self._records

    def _save(self) -> None:
        """原子写入 JSON 文件。"""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(".json.tmp")
            tmp_path.write_text(
                json.dumps(self._records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, self._path)
        except Exception as exc:
            logger.warning(f"[group_watch] 保存静默记录文件失败: {exc}")

    def get_silence(self, session_id: str) -> float:
        now = time.time()
        until = self._records.get(session_id, 0.0)
        if until > now:
            return until
        if session_id in self._records:
            self._records.pop(session_id, None)
            self._save()
        return 0.0

    def set_silence(self, session_id: str, until: float) -> None:
        self._records[session_id] = float(until)
        self._save()

    def clear_silence(self, session_id: str) -> None:
        if session_id in self._records:
            self._records.pop(session_id, None)
            self._save()


class StateManager:
    """管理群聊语境缓冲区、会话状态与生命周期。"""

    def __init__(self) -> None:
        self._buffers: dict[str, list[dict[str, Any]]] = {}
        self._states: dict[str, dict[str, Any]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._debounce_tasks: dict[str, asyncio.Task] = {}
        self.silence_storage = SilenceStorage()

    @property
    def buffers(self) -> dict[str, list[dict[str, Any]]]:
        return self._buffers

    @property
    def states(self) -> dict[str, dict[str, Any]]:
        return self._states

    @property
    def debounce_tasks(self) -> dict[str, asyncio.Task]:
        return self._debounce_tasks

    def get_state(self, session_id: str, bot_id: str = "", bot_name: str = "") -> dict[str, Any]:
        state = self._states.get(session_id)
        if state is None:
            persisted_silence = self.silence_storage.get_silence(session_id)
            state = {
                "last_human": 0.0,
                "last_bot": 0.0,
                "follow_until": 0.0,
                "follow_from": 0.0,
                "follow_target_uid": "",
                "follow_target_name": "",
                "silence_until": persisted_silence,
                "last_repeated_text": "",
                "pending_follow_eval": False,
                "unrelated": 0,
                "last_human_event": None,
                "decisions": [],
                "bot_id": bot_id,
                "bot_name": bot_name,
            }
            self._states[session_id] = state
        return state

    def set_silence(self, session_id: str, until: float) -> None:
        state = self.get_state(session_id)
        state["silence_until"] = until
        self.silence_storage.set_silence(session_id, until)

    def clear_silence(self, session_id: str) -> None:
        state = self.get_state(session_id)
        state["silence_until"] = 0.0
        self.silence_storage.clear_silence(session_id)

    def get_lock(self, session_id: str) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    def append(self, session_id: str, entry: dict[str, Any]) -> None:
        buf = self._buffers.setdefault(session_id, [])
        buf.append(entry)
        if len(buf) > BUFFER_LIMIT:
            del buf[: len(buf) - BUFFER_LIMIT]

    def render_context(self, session_id: str, count: int, bot_id: str, bot_name: str) -> str:
        buf = self._buffers.get(session_id) or []
        if not buf:
            return ""

        slice_items = buf[-max(1, count):]
        display_bot_name = bot_name or "我"
        lines: list[str] = []
        for item in slice_items:
            stamp = datetime.fromtimestamp(item["ts"]).strftime("%H:%M")
            if item.get("is_bot"):
                who = f"{display_bot_name}({bot_id})" if bot_id else display_bot_name
            else:
                name = str(item.get("name") or "").strip()
                uid = str(item.get("uid") or "?")
                who = f"{name}({uid})" if name else uid
            line = f"[{stamp}] {who}"

            quote = item.get("quote") or {}
            if quote:
                q_name = str(quote.get("name") or "").strip()
                q_uid = str(quote.get("uid") or "?")
                q_who = f"{q_name}({q_uid})" if q_name else q_uid
                quoted = str(quote.get("text") or "")[:60]
                line += f" → 回复 {q_who}「{quoted}」"

            at_targets = item.get("at_targets") or []
            ats = item.get("ats") or []
            if at_targets:
                targets = []
                for a_item in at_targets:
                    a_uid = str(a_item.get("uid") or "")
                    a_name = str(a_item.get("name") or "").strip()
                    if bot_id and a_uid == str(bot_id):
                        targets.append(f"{display_bot_name}({bot_id})" if bot_id else display_bot_name)
                    elif a_name and a_uid:
                        targets.append(f"{a_name}({a_uid})")
                    else:
                        targets.append(a_name or a_uid)
                if targets:
                    line += " → @" + ",".join(targets)
            elif ats:
                targets = []
                for a in ats:
                    if bot_id and str(a) == str(bot_id):
                        targets.append(f"{display_bot_name}({bot_id})" if bot_id else display_bot_name)
                    else:
                        targets.append(str(a))
                if targets:
                    line += " → @" + ",".join(targets)

            text = str(item.get("text", ""))
            desc = str(item.get("image_desc") or "").strip()
            if desc:
                text = text.replace("[图片]", f"[图片：{desc}]", 1)
            line += f": {text}"
            lines.append(line)

        text = "\n".join(lines)
        if len(text) > 4000:
            text = text[-4000:]
        return text

    def get_participants(self, session_id: str, count: int = 40) -> list[str]:
        seen: list[str] = []
        for item in (self._buffers.get(session_id) or [])[-count:]:
            uid = str(item.get("uid") or "")
            name = str(item.get("name") or "").strip()
            label = f"{name}({uid})" if name else uid
            if uid and not item.get("is_bot") and label not in seen:
                seen.append(label)
        return seen

    def record_decision(
        self,
        state: dict[str, Any],
        *,
        mode: str,
        related: bool,
        target: str = "everyone",
        addressed: bool,
        reply: bool,
        topic: str = "active",
        reason: str,
        sent: bool,
        score: int = 0,
        threshold: int = 0,
        target_text: str = "",
        target_uid: str = "",
        target_name: str = "",
    ) -> None:
        decisions = state.setdefault("decisions", [])
        decisions.append(
            {
                "ts": time.time(),
                "mode": mode,
                "related": related,
                "target": target,
                "addressed": addressed,
                "reply": reply,
                "score": score,
                "threshold": threshold,
                "topic_state": topic,
                "reason": reason,
                "sent": sent,
                "target_text": target_text,
                "target_uid": target_uid,
                "target_name": target_name,
            }
        )
        if len(decisions) > MAX_DECISIONS:
            del decisions[: len(decisions) - MAX_DECISIONS]

    def maybe_cleanup_sessions(self) -> None:
        """限制活跃会话数量，淘汰超过 24 小时未发言的僵尸群聊，防止内存泄漏。"""
        if len(self._states) <= 100:
            return
        now = time.time()
        stale = [
            sid
            for sid, s in self._states.items()
            if (now - s.get("last_human", 0.0) > 86400)
            and (now >= s.get("follow_until", 0.0))
            and (now >= s.get("silence_until", 0.0))
        ]
        for sid in stale:
            self._states.pop(sid, None)
            self._buffers.pop(sid, None)
            self._locks.pop(sid, None)
            task = self._debounce_tasks.pop(sid, None)
            if task and not task.done():
                task.cancel()

    def cancel_all_debounce(self) -> None:
        for task in self._debounce_tasks.values():
            if not task.done():
                task.cancel()
        self._debounce_tasks.clear()
