from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.platform import MessageType
from astrbot.api.star import Context, Star
from astrbot.core.message.components import At, AtAll, Image, Plain, Reply

try:
    from .constants import (
        DEFAULT_JUDGE_PROVIDER,
        DEFAULT_JUDGE_SYSTEM_PROMPT,
        IMAGE_WAIT_SECONDS,
        WILLINGNESS_LABELS,
        WILLINGNESS_THRESHOLDS,
        as_bool,
    )
    from .image_service import ImageService
    from .state import StateManager
except (ImportError, ValueError):
    from constants import (
        DEFAULT_JUDGE_PROVIDER,
        DEFAULT_JUDGE_SYSTEM_PROMPT,
        IMAGE_WAIT_SECONDS,
        WILLINGNESS_LABELS,
        WILLINGNESS_THRESHOLDS,
        as_bool,
    )
    from image_service import ImageService
    from state import StateManager


class GroupWatchPlugin(Star):
    """群聊巡视：直接全流通路接入轻量判断模型；
    支持发言对象(target)识别与群友对话一票否决；
    支持跟进期多轮互动倾斜、关注词加权与 @ 闭嘴 持久化熔断静默。"""

    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self.config = config if config is not None else {}
        self.state_mgr = StateManager()
        self.img_service = ImageService()
        self._eval_semaphore = asyncio.Semaphore(3)
        self._enabled = bool(self._get("enabled", True))
        self._bot_id = ""
        self._bot_name = ""

    # ------------------------------------------------------------------
    # 配置读取与模型选项同步
    # ------------------------------------------------------------------
    def _get(self, key: str, default: Any = None) -> Any:
        try:
            value = self.config.get(key, default)
        except Exception:
            value = default
        return default if value is None else value

    def _int(self, key: str, default: int) -> int:
        try:
            return int(self._get(key, default))
        except Exception:
            return default

    def _str(self, key: str, default: str = "") -> str:
        value = self._get(key, default)
        return str(value) if value is not None else default

    def _sync_judge_options(self) -> None:
        """把 AstrBot 当前可用的 chat 模型写进设置面板的下拉列表（只读环境安全跳过）。"""
        try:
            options: list[str] = []
            for provider in self.context.get_all_providers() or []:
                cfg = getattr(provider, "provider_config", {}) or {}
                pid = str(cfg.get("id") or "").strip()
                if not pid:
                    continue
                try:
                    raw_type = provider.meta().provider_type
                    ptype = str(getattr(raw_type, "value", raw_type) or "")
                except Exception:
                    ptype = "chat_completion"
                if ptype and ptype != "chat_completion":
                    continue
                if pid not in options:
                    options.append(pid)
            if not options:
                return
            path = Path(__file__).resolve().with_name("_conf_schema.json")
            if not path.is_file() or not os.access(path, os.W_OK):
                return
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            field = data.get("judge_provider_id")
            if not isinstance(field, dict) or field.get("options") == options:
                return
            field["options"] = options
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, path)
            logger.info(f"[group_watch] 判断模型列表已更新: {options}")
        except (PermissionError, OSError) as exc:
            logger.debug(f"[group_watch] 运行环境为只读，跳过更新配置文件列表: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[group_watch] 更新判断模型列表失败: {exc}")

    def _allowed(self, session_id: str, group_id: str) -> bool:
        groups = self._get("groups", []) or []
        if isinstance(groups, str):
            groups = [part.strip() for part in re.split(r"[,;\n]", groups)]
        groups = [str(item).strip() for item in groups if str(item).strip()]
        if not groups:
            return True
        for item in groups:
            if item == session_id:
                return True
            if group_id and (item == group_id or item.endswith(":" + group_id)):
                return True
        return False

    @staticmethod
    def _extract_silence_minutes(text: str, default_min: int) -> int:
        """从闭嘴文本中提取自定义静默分钟数（如“闭嘴 30”、“闭嘴半小时”、“闭嘴 2小时”），未指定则使用配置默认值。"""
        if "闭嘴半小时" in text or "闭嘴半个小时" in text:
            return 30
        m_hour = re.search(r"闭嘴\s*(\d+(?:\.\d+)?)\s*(?:个?小时|h|hr)", text, re.I)
        if m_hour:
            try:
                val = float(m_hour.group(1))
                if val > 0:
                    return max(1, int(val * 60))
            except Exception:
                pass
        m_min = re.search(r"闭嘴\s*(\d+)(?:\s*(?:分钟|分|m|min))?", text, re.I)
        if m_min:
            try:
                val = int(m_min.group(1))
                if val > 0:
                    return max(1, val)
            except Exception:
                pass
        return default_min

    def _threshold(self) -> int:
        willingness = self._int("reply_willingness", 3)
        return WILLINGNESS_THRESHOLDS.get(willingness, 55)

    def _debounce(self) -> int:
        return max(0, self._int("debounce_seconds", 3))

    def _wake_words(self) -> list[str]:
        raw = self._get("wake_words", [])
        words: list[str] = []
        if isinstance(raw, list):
            for item in raw:
                s = str(item or "").strip()
                if s and s not in words:
                    words.append(s)
        elif isinstance(raw, str):
            for part in re.split(r"[,，\n|]+", raw):
                s = part.strip()
                if s and s not in words:
                    words.append(s)
        return words

    def _identity_block(self, session_id: str = "") -> str:
        bot_id = self._bot_id
        bot_name = self._bot_name or "机器人"
        participants = (
            self.state_mgr.get_participants(session_id, count=15) if session_id else []
        )
        part_str = "、".join(participants[:12]) if participants else "暂无"

        return (
            f"【自身身份】：你在本群的唯一名字是「{bot_name}」"
            + (f"，账号是 {bot_id}" if bot_id else "")
            + f"；“我”仅指你自己。\n"
            f"【群内近期活跃群友】：{part_str}\n"
            f"【关键人称与归属判定铁律】：\n"
            f"1. 其他群友称呼与外号辨析：群内其他群友有各自的全名、简称与外号。群友在消息中叫他人名字或外号，是在跟该群友说话，绝不是在跟你说话！\n"
            f"2. 第二人称“你”的归属：群友说“你”时，若上文中其他人类群友正在互相接话讨论，这个“你”指向的是那位人类群友，绝对不是你！即便你刚才插过话，也不要误认为所有“你”都是对自己说的！\n"
            f"3. 只有明确提到「{bot_name}」、显式 @你，或明确针对机器人刚才发表的观点进行直接互动时，target 才能判定为 \"bot\"。"
        )

    # ------------------------------------------------------------------
    # 消息解析与辅助
    # ------------------------------------------------------------------
    def _parse_event(self, event: AstrMessageEvent) -> dict[str, Any]:
        parts: list[str] = []
        ats: list[str] = []
        at_targets: list[dict[str, str]] = []
        quote: dict[str, Any] | None = None
        has_image = False
        has_at_all = False
        for comp in event.get_messages():
            if isinstance(comp, Reply):
                quoted = getattr(comp, "message_str", "") or ""
                if not quoted and getattr(comp, "chain", None):
                    quoted = "".join(
                        getattr(c, "text", "") or "" for c in comp.chain or []
                    )
                quote_uid = str(getattr(comp, "sender_id", "") or "")
                quote_name = str(
                    getattr(comp, "sender_nickname", "")
                    or getattr(comp, "sender_name", "")
                    or ""
                )
                quote = {
                    "uid": quote_uid,
                    "name": quote_name,
                    "text": re.sub(r"\s+", " ", str(quoted)).strip()[:60],
                }
                continue
            if isinstance(comp, At):
                qq = str(getattr(comp, "qq", "") or "")
                name = str(getattr(comp, "name", "") or "")
                if isinstance(comp, AtAll) or qq == "all":
                    has_at_all = True
                if qq:
                    ats.append(qq)
                    at_targets.append({"uid": qq, "name": name})
                at_label = f"{name}({qq})" if (name and qq) else (name or qq)
                parts.append(f"@{at_label}")
                continue
            text = getattr(comp, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
                continue
            cls = comp.__class__.__name__.lower()
            if isinstance(comp, Image) or "image" in cls:
                has_image = True
                parts.append("[图片]")
            elif "record" in cls or "audio" in cls:
                parts.append("[语音]")
            elif "video" in cls:
                parts.append("[视频]")
            elif "file" in cls:
                parts.append("[文件]")

        sender_name = ""
        try:
            sender_name = str(event.get_sender_name() or "").strip()
        except Exception:
            sender_name = ""

        return {
            "text": re.sub(r"\s+", " ", "".join(parts)).strip(),
            "sender_name": sender_name,
            "ats": ats,
            "at_targets": at_targets,
            "quote": quote,
            "has_image": has_image,
            "has_at_all": has_at_all,
        }

    def _is_at_me(self, parsed: dict[str, Any], state: dict[str, Any]) -> bool:
        bot_id = self._bot_id or state.get("bot_id") or ""
        if bot_id and bot_id in parsed.get("ats", []):
            return True
        quote = parsed.get("quote") or {}
        return bool(bot_id and str(quote.get("uid") or "") == bot_id)

    async def _send_plain_message(
        self, event: AstrMessageEvent, session_id: str, text: str
    ) -> bool:
        try:
            chain = MessageChain([Plain(text)])
            if hasattr(event, "send"):
                try:
                    await event.send(chain)
                    return True
                except Exception as exc:
                    logger.debug(
                        f"[group_watch] event.send 失败，回退 context.send_message: {exc}"
                    )
            if hasattr(self.context, "send_message"):
                await self.context.send_message(session_id, chain)
                return True
            return False
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[group_watch] 发送纯文本消息失败: {exc}", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # 图片摘要拦截
    # ------------------------------------------------------------------
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: Any) -> None:
        if not as_bool(self._get("image_memory", True), True):
            return
        entry = event.get_extra("_gw_image_entry")
        if not entry:
            return

        session_id = event.unified_msg_origin
        images = self.img_service.extract_prompt_images(req)
        if not images:
            return

        provider = self.context.get_using_provider(umo=session_id)
        limit = max(40, self._int("image_summary_chars", 200))
        self.img_service.create_describe_task(
            provider, session_id, entry, images, char_limit=limit
        )

    # ------------------------------------------------------------------
    # 消息事件监听与分发
    # ------------------------------------------------------------------
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=997)
    async def on_group_message(self, event: AstrMessageEvent) -> None:
        if event.get_extra("_gw_triggered"):
            return

        self.state_mgr.maybe_cleanup_sessions()

        session_id = event.unified_msg_origin
        group_id = str(event.get_group_id() or "")
        if not self._allowed(session_id, group_id):
            return

        sender_id = str(event.get_sender_id() or "")
        if not self._bot_id:
            self._bot_id = str(event.get_self_id() or "")
        if not self._bot_name:
            try:
                self._bot_name = str(event.get_self_name() or "").strip()
            except Exception:
                self._bot_name = ""

        if sender_id and sender_id == self._bot_id:
            return

        parsed = self._parse_event(event)
        if not parsed["text"] and not parsed["ats"] and not parsed["quote"]:
            return

        sender_name = parsed["sender_name"]
        now = time.time()
        state = self.state_mgr.get_state(session_id, self._bot_id, self._bot_name)
        state["bot_id"] = self._bot_id
        if self._bot_name:
            state["bot_name"] = self._bot_name

        if state["follow_until"] and now >= state["follow_until"]:
            state["follow_until"] = 0.0
            state["follow_from"] = 0.0
            state["follow_target_uid"] = ""
            state["follow_target_name"] = ""
            state["unrelated"] = 0

        entry = {
            "ts": now,
            "uid": sender_id,
            "name": sender_name,
            "text": parsed["text"],
            "is_bot": False,
            "ats": parsed["ats"],
            "at_targets": parsed.get("at_targets", []),
            "quote": parsed["quote"],
            "has_image": parsed["has_image"],
        }
        self.state_mgr.append(session_id, entry)
        state["last_human"] = now
        state["last_human_event"] = event

        if not self._enabled:
            return

        # -------------------------------------------------------------
        # 复读机彩蛋
        # -------------------------------------------------------------
        repeat_target = max(0, self._int("repeat_trigger_count", 3))
        if (
            repeat_target >= 2
            and parsed["text"]
            and not parsed["text"].strip().startswith("/")
            and not parsed["has_image"]
        ):
            clean_text = parsed["text"].strip()
            if 1 <= len(clean_text) <= 120:
                buf = self.state_mgr.buffers.get(session_id) or []
                consecutive_count = 0
                for item in reversed(buf):
                    if item.get("is_bot"):
                        break
                    if str(item.get("text") or "").strip() == clean_text:
                        consecutive_count += 1
                    else:
                        break
                if (
                    consecutive_count >= repeat_target
                    and state.get("last_repeated_text") != clean_text
                ):
                    state["last_repeated_text"] = clean_text
                    sent = await self._send_plain_message(
                        event, session_id, clean_text
                    )
                    if sent:
                        rep_now = time.time()
                        self.state_mgr.append(
                            session_id,
                            {
                                "ts": rep_now,
                                "uid": self._bot_id or "bot",
                                "name": self._bot_name or "我",
                                "text": clean_text,
                                "is_bot": True,
                                "ats": [],
                                "quote": None,
                            },
                        )
                        state["last_bot"] = rep_now
                        self.state_mgr.record_decision(
                            state,
                            mode="repeat",
                            related=True,
                            target="everyone",
                            addressed=False,
                            reply=True,
                            topic="active",
                            reason=f"触发排队复读跟读（群友已连续发送 {consecutive_count} 次）",
                            sent=True,
                            score=100,
                            threshold=0,
                            target_text=clean_text,
                            target_uid=sender_id,
                            target_name=sender_name,
                        )
                        logger.info(
                            f"[group_watch] {session_id} 触发排队复读，原样跟读: {clean_text}"
                        )
                        return

        # 含图片的消息不主动插话：只缓存，交由主流程
        if parsed["has_image"]:
            self.state_mgr.record_decision(
                state,
                mode="image",
                related=False,
                target="everyone",
                addressed=False,
                reply=False,
                topic="active",
                reason="包含图片消息，已存入图文语境缓冲区，交由主流程处理",
                sent=False,
                score=0,
                threshold=0,
                target_text=parsed["text"],
                target_uid=sender_id,
                target_name=sender_name,
            )
            if self._is_at_me(parsed, state) or now < state["follow_until"]:
                event.set_extra("_gw_image_entry", entry)
            return

        # 指令与 @全体 让给主流程
        if parsed["text"].strip().startswith("/") or parsed["has_at_all"]:
            self.state_mgr.record_decision(
                state,
                mode="command",
                related=False,
                target="everyone",
                addressed=False,
                reply=False,
                topic="active",
                reason="指令或 @全体 消息，让给主流程处理，巡视不抢话",
                sent=False,
                score=0,
                threshold=0,
                target_text=parsed["text"],
                target_uid=sender_id,
                target_name=sender_name,
            )
            return

        if self._is_at_me(parsed, state):
            # 唯一静默触发条件：群友显式 @ 机器人且文本包含“闭嘴”
            if "闭嘴" in parsed["text"]:
                silence_min = self._extract_silence_minutes(
                    parsed["text"], max(1, self._int("silence_duration_minutes", 10))
                )
                silence_until = now + silence_min * 60
                self.state_mgr.set_silence(session_id, silence_until)
                state["follow_until"] = 0.0
                state["follow_from"] = 0.0
                state["follow_target_uid"] = ""
                state["follow_target_name"] = ""
                state["unrelated"] = 0
                logger.info(
                    f"[group_watch] {session_id} 群友显式 @ 机器人且要求闭嘴，持久化静默 {silence_min} 分钟"
                )
                self.state_mgr.record_decision(
                    state,
                    mode="silence",
                    related=False,
                    target="bot",
                    addressed=True,
                    reply=False,
                    topic="ended",
                    reason=f"群友显式 @ 机器人且要求闭嘴，进入静默期 {silence_min} 分钟",
                    sent=False,
                    score=0,
                    threshold=0,
                    target_text=parsed["text"],
                    target_uid=sender_id,
                    target_name=sender_name,
                )
                event.stop_event()
                return

            # 群友显式 @ 机器人且未要求闭嘴：若处于静默期，则提前解除静默
            if state.get("silence_until"):
                self.state_mgr.clear_silence(session_id)
                logger.info(
                    f"[group_watch] {session_id} 群友显式 @ 机器人，提前解除静默期"
                )

            # 开启跟进窗口
            if as_bool(self._get("follow_enabled", True), True):
                state["follow_until"] = (
                    now + max(1, self._int("follow_max_minutes", 10)) * 60
                )
                state["follow_from"] = now
                state["follow_target_uid"] = sender_id
                state["follow_target_name"] = sender_name
                state["unrelated"] = 0

            self.state_mgr.record_decision(
                state,
                mode="wake",
                related=True,
                target="bot",
                addressed=True,
                reply=True,
                topic="active",
                reason="群友显式 @ 机器人，交由主模型直接回复",
                sent=True,
                score=100,
                threshold=0,
                target_text=parsed["text"],
                target_uid=sender_id,
                target_name=sender_name,
            )
            return

        # -------------------------------------------------------------
        # 关注关键词/唤起词与控制词检测
        # -------------------------------------------------------------
        clean_text = parsed["text"].strip()
        matched_wake_word = ""
        wake_words = self._wake_words()
        bot_name = self._bot_name or "机器人"
        if clean_text and not clean_text.startswith("/"):
            text_lower = clean_text.lower()
            if "闭嘴" in clean_text and (
                self._is_at_me(parsed, state) or bot_name.lower() in text_lower
            ):
                silence_min = self._extract_silence_minutes(
                    clean_text, max(1, self._int("silence_duration_minutes", 10))
                )
                silence_until = now + silence_min * 60
                self.state_mgr.set_silence(session_id, silence_until)
                state["follow_until"] = 0.0
                state["follow_from"] = 0.0
                state["follow_target_uid"] = ""
                state["follow_target_name"] = ""
                state["unrelated"] = 0
                logger.info(
                    f"[group_watch] {session_id} 收到闭嘴指令，持久化静默 {silence_min} 分钟"
                )
                self.state_mgr.record_decision(
                    state,
                    mode="silence",
                    related=False,
                    target="bot",
                    addressed=True,
                    reply=False,
                    topic="ended",
                    reason=f"群友要求闭嘴，进入静默期 {silence_min} 分钟",
                    sent=False,
                    score=0,
                    threshold=0,
                    target_text=parsed["text"],
                    target_uid=sender_id,
                    target_name=sender_name,
                )
                event.stop_event()
                return

            for w in wake_words:
                if w.lower() in text_lower:
                    matched_wake_word = w
                    break

        if matched_wake_word:
            entry["matched_kw"] = matched_wake_word

        # 若处于静默期，保持静默不主动插话
        if state.get("silence_until") and now < state["silence_until"]:
            left_sec = int(state["silence_until"] - now)
            self.state_mgr.record_decision(
                state,
                mode="silence_skip",
                related=False,
                target="everyone",
                addressed=False,
                reply=False,
                topic="silence",
                reason=f"当前处于静默期（剩余 {left_sec}s），保持静默不插话",
                sent=False,
                score=0,
                threshold=0,
                target_text=parsed["text"],
                target_uid=sender_id,
                target_name=sender_name,
            )
            return

        follow_enabled = as_bool(self._get("follow_enabled", True), True)
        if follow_enabled and now < state["follow_until"]:
            self._schedule(session_id, "follow", delay=self._debounce(), entry=entry)
        elif as_bool(self._get("judge_enabled", True), True):
            self._schedule(session_id, "judge", delay=self._debounce(), entry=entry)

    # ------------------------------------------------------------------
    # 机器人消息与发言记录
    # ------------------------------------------------------------------
    @filter.after_message_sent()
    async def on_bot_message_sent(self, event: AstrMessageEvent) -> None:
        try:
            if event.get_message_type() != MessageType.GROUP_MESSAGE:
                return
            session_id = event.unified_msg_origin
            group_id = str(event.get_group_id() or "")
            if not self._allowed(session_id, group_id):
                return
            now = time.time()
            state = self.state_mgr.get_state(session_id, self._bot_id, self._bot_name)
            old_follow_from = float(state.get("follow_from") or 0.0)
            state["last_bot"] = now
            state["follow_from"] = 0.0
            state["generating_until"] = 0.0

            if state.get("pending_follow_eval"):
                state["pending_follow_eval"] = False
                if state["last_human"] > old_follow_from:
                    self._schedule(session_id, "follow", delay=1)

            result = event.get_result()
            if result is None or not result.is_model_result():
                return
            text = ""
            try:
                text = result.get_plain_text().strip()
            except Exception:
                text = ""
            if not text:
                return
            buf = self.state_mgr.buffers.setdefault(session_id, [])
            if (
                buf
                and buf[-1].get("is_bot")
                and buf[-1].get("text") == text
                and now - buf[-1]["ts"] < 5
            ):
                return
            self.state_mgr.append(
                session_id,
                {
                    "ts": now,
                    "uid": self._bot_id or "bot",
                    "name": self._bot_name or "我",
                    "text": text,
                    "is_bot": True,
                    "ats": [],
                    "quote": None,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[group_watch] 记录机器人消息失败: {exc}")

    def record_bot_message(self, session_id: str, text: str) -> None:
        """供外部插件（如 aichat_bridge）通知机器人发言。"""
        try:
            now = time.time()
            state = self.state_mgr.get_state(session_id, self._bot_id, self._bot_name)
            old_follow_from = float(state.get("follow_from") or 0.0)
            state["last_bot"] = now
            state["follow_from"] = 0.0
            state["generating_until"] = 0.0

            if state.get("pending_follow_eval"):
                state["pending_follow_eval"] = False
                if state["last_human"] > old_follow_from:
                    self._schedule(session_id, "follow", delay=1)

            clean_text = str(text or "").strip()
            if not clean_text:
                return
            buf = self.state_mgr.buffers.setdefault(session_id, [])
            if (
                buf
                and buf[-1].get("is_bot")
                and buf[-1].get("text") == clean_text
                and now - buf[-1]["ts"] < 5
            ):
                return
            self.state_mgr.append(
                session_id,
                {
                    "ts": now,
                    "uid": self._bot_id or "bot",
                    "name": self._bot_name or "我",
                    "text": clean_text,
                    "is_bot": True,
                    "ats": [],
                    "quote": None,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[group_watch] record_bot_message 异常: {exc}")

    # ------------------------------------------------------------------
    # 防抖调度
    # ------------------------------------------------------------------
    def _schedule(
        self,
        session_id: str,
        kind: str,
        delay: int | None = None,
        entry: dict[str, Any] | None = None,
    ) -> None:
        state = self.state_mgr.get_state(session_id, self._bot_id, self._bot_name)
        if entry:
            pending = state.setdefault("_pending_entries", [])
            pending.append(entry)

        wait_sec = self._debounce() if delay is None else max(0, delay)
        old_task = self.state_mgr.debounce_tasks.get(session_id)
        if old_task and not old_task.done():
            old_task.cancel()
        self.state_mgr.debounce_tasks[session_id] = asyncio.create_task(
            self._debounced_job(session_id, kind, wait_sec)
        )

    async def _debounced_job(self, session_id: str, kind: str, wait_sec: int) -> None:
        try:
            if wait_sec > 0:
                await asyncio.sleep(wait_sec)
            await self._evaluate(session_id, kind)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[group_watch] 评估任务异常: {exc}", exc_info=True)

    # ------------------------------------------------------------------
    # 判定与评估流程
    # ------------------------------------------------------------------
    async def _evaluate(self, session_id: str, kind: str) -> None:
        # 并发信号量控制：若已有 3 个评估在飞，直接放弃本次主动接话评估，不积压任务
        if self._eval_semaphore.locked():
            logger.debug(
                f"[group_watch] 当前已有 3 个模型评估正在运行，跳过本次评估 ({session_id})"
            )
            return

        lock = self.state_mgr.get_lock(session_id)
        if lock.locked():
            return
        async with lock:
            if not self._enabled:
                return
            state = self.state_mgr.get_state(session_id, self._bot_id, self._bot_name)
            buf = self.state_mgr.buffers.get(session_id) or []
            if not buf:
                return
            now = time.time()

            pending = state.pop("_pending_entries", [])
            threshold = self._threshold()
            if len(pending) > 1:
                for item in pending[:-1]:
                    self.state_mgr.record_decision(
                        state,
                        mode="merged",
                        related=False,
                        target="everyone",
                        addressed=False,
                        reply=False,
                        topic="active",
                        reason="连续快速发言，已合并至下句一同评估",
                        sent=False,
                        score=0,
                        threshold=threshold,
                        target_text=str(item.get("text") or ""),
                        target_uid=str(item.get("uid") or ""),
                        target_name=str(item.get("name") or ""),
                    )

            if state.get("silence_until") and now < state["silence_until"]:
                state["last_human_event"] = None
                return

            if now < float(state.get("generating_until") or 0.0):
                logger.debug(
                    f"[group_watch] {session_id} 主模型正在生成回复中，暂缓并发评估"
                )
                state["pending_follow_eval"] = True
                return

            follow_enabled = as_bool(self._get("follow_enabled", True), True)
            follow_active = follow_enabled and (now < state["follow_until"])
            mode = "follow" if follow_active else "judge"

            if mode == "follow":
                follow_from = float(state.get("follow_from") or 0.0)
                if follow_from:
                    if state["last_bot"] < follow_from:
                        if now - follow_from < 30:
                            state["pending_follow_eval"] = True
                            logger.debug(
                                f"[group_watch] {session_id} 主流程正在回复中，标记待回复后跟进（{kind}）"
                            )
                            return
                        logger.debug(
                            f"[group_watch] {session_id} 等待主回复超时（>30s），自动解除等待"
                        )
                    state["follow_from"] = 0.0

            last_human_text = ""
            last_human_uid = ""
            last_human_name = ""
            for item in reversed(buf):
                if not item.get("is_bot"):
                    last_human_text = str(item.get("text") or "")
                    last_human_uid = str(item.get("uid") or "")
                    last_human_name = str(item.get("name") or "")
                    break

            if not last_human_text and not last_human_uid:
                state["last_human_event"] = None
                return

            await self.img_service.wait_images(
                session_id,
                self._int("judge_context_count", 8),
                self.state_mgr.buffers,
                IMAGE_WAIT_SECONDS,
            )

            # 在信号量保护下调用模型评估
            async with self._eval_semaphore:
                judge = await self._judge_related(session_id, last_human_uid)

            target = str(judge.get("target") or "everyone").lower()
            score = int(judge.get("score") or 0)
            threshold = int(judge.get("threshold") or 0)
            reason = str(judge.get("reason") or "")

            related = score >= threshold

            if mode == "follow":
                if not related:
                    state["unrelated"] += 1
                    if state["unrelated"] >= max(
                        1, self._int("follow_exit_unrelated", 5)
                    ):
                        state["follow_until"] = 0.0
                        state["follow_from"] = 0.0
                        state["follow_target_uid"] = ""
                        state["follow_target_name"] = ""
                        state["unrelated"] = 0
                        logger.info(
                            f"[group_watch] {session_id} 连续无关，结束跟进并清空互动标记"
                        )
                else:
                    state["unrelated"] = 0

            sent = False
            if related:
                last_event = state.get("last_human_event")
                if last_event:
                    state["generating_until"] = now + 15
                    sent = await self._wake_main_pipeline(last_event)
                    if sent:
                        if as_bool(self._get("follow_enabled", True), True):
                            state["follow_until"] = (
                                now + max(1, self._int("follow_max_minutes", 10)) * 60
                            )
                            state["follow_from"] = now
                            state["follow_target_uid"] = last_human_uid
                            state["follow_target_name"] = last_human_name
                            state["unrelated"] = 0

            state["last_human_event"] = None

            self.state_mgr.record_decision(
                state,
                mode=mode,
                related=related,
                target=target,
                addressed=False,
                reply=related,
                topic="active",
                reason=reason,
                sent=sent,
                score=score,
                threshold=threshold,
                target_text=last_human_text,
                target_uid=last_human_uid,
                target_name=last_human_name,
            )

    async def _wake_main_pipeline(self, last_event: AstrMessageEvent) -> bool:
        """唤醒 AstrBot 主流程，将当前群消息交由主模型回复。"""
        try:
            platform_name = last_event.get_platform_name()
            platforms = self.context.platform_manager.get_insts()
            adapter = next(
                (
                    p
                    for p in platforms
                    if p.meta().name == platform_name or p.meta().id == platform_name
                ),
                None,
            )
            if adapter is None and platforms:
                adapter = platforms[0]
            if adapter is None:
                logger.warning(f"[group_watch] 找不到平台适配器: {platform_name}")
                return False

            bot_id = self._bot_id or str(last_event.get_self_id() or "")
            msg_obj = copy.copy(last_event.message_obj)
            orig_msgs = list(getattr(msg_obj, "message", []) or [])
            qq_val = int(bot_id) if bot_id.isdigit() else bot_id

            msg_obj.message = [At(qq=qq_val)] + [
                m
                for m in orig_msgs
                if not (
                    isinstance(m, At) and str(getattr(m, "qq", "")) == str(bot_id)
                )
            ]

            wake_event = adapter.create_event(msg_obj)
            wake_event.is_wake = True
            wake_event.is_at_or_wake_command = True
            wake_event.set_extra("_gw_triggered", True)
            adapter.commit_event(wake_event)
            logger.info(
                f"[group_watch] 判定接话，已唤醒主模型回复 (平台: {adapter.meta().name})"
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[group_watch] 唤醒主流程异常: {exc}", exc_info=True)
            return False

    async def _judge_related(
        self, session_id: str, sender_id: str = ""
    ) -> dict[str, Any]:
        provider = self._judge_provider(session_id)
        if provider is None:
            return {
                "related": False,
                "target": "everyone",
                "score": 0,
                "raw_score": 0,
                "threshold": 0,
                "reason": "没有可用的判断模型",
            }
        context = self.state_mgr.render_context(
            session_id,
            self._int("judge_context_count", 8),
            self._bot_id,
            self._bot_name,
        )
        state = self.state_mgr.get_state(session_id, self._bot_id, self._bot_name)
        threshold = self._threshold()

        raw_prompt = self._str("judge_system_prompt", "").strip()
        if not raw_prompt:
            raw_prompt = DEFAULT_JUDGE_SYSTEM_PROMPT
        if "{identity_block}" in raw_prompt:
            system = raw_prompt.replace(
                "{identity_block}", self._identity_block(session_id)
            )
        else:
            system = self._identity_block(session_id) + "\n" + raw_prompt

        buf = self.state_mgr.buffers.get(session_id) or []
        turn_hint = ""
        human_buf = [b for b in buf if not b.get("is_bot")]
        if len(human_buf) >= 2:
            prev_msg = human_buf[-2]
            curr_msg = human_buf[-1]
            prev_uid = str(prev_msg.get("uid") or "")
            prev_name = str(prev_msg.get("name") or "")
            curr_uid = str(curr_msg.get("uid") or "")
            curr_name = str(curr_msg.get("name") or "")

            prev_ats = [str(a) for a in prev_msg.get("ats") or []]
            prev_quote_uid = str((prev_msg.get("quote") or {}).get("uid") or "")
            if curr_uid and (curr_uid in prev_ats or curr_uid == prev_quote_uid):
                turn_hint = (
                    f"\n[上下文提示] 上一句是群友 {prev_name}({prev_uid}) 定向向当前发言者 {curr_name}({curr_uid}) 发言，"
                    f"当前发言者正在回应对方。若涉及两人私人专属事务请压低评分（建议 ≤35分），若属于群内公开讨论/开放话题可综合评估。\n"
                )

        now = time.time()
        in_follow = now < float(state.get("follow_until") or 0.0)
        follow_uid = str(state.get("follow_target_uid") or "")
        follow_name = str(state.get("follow_target_name") or "")
        is_target_speaker = bool(in_follow and follow_uid and sender_id == follow_uid)

        context_hint = ""
        if is_target_speaker:
            target_display = (
                f"{follow_name}({follow_uid})" if follow_name else follow_uid
            )
            context_hint = (
                f"\n[多轮互动提示] 机器人刚才正在与群友 {target_display} 连续对话中。"
                f"如果该群友是在承接刚才与你的话题进行追问/调侃，请判定为 \"bot\"；"
                f"【注意】：若该群友转向跟其他群友说话或在群里自言自语，必须如实判定为 \"user\" 或 \"everyone\"，严禁误判为 bot！\n"
            )

        last_human_text = str((human_buf[-1] if human_buf else {}).get("text") or "")
        matched_kws = [
            w for w in self._wake_words() if w.lower() in last_human_text.lower()
        ]
        kw_hint = ""
        if matched_kws:
            kw_str = "、".join(matched_kws)
            kw_hint = (
                f"\n[关注关键词命中提醒]：当前消息中出现了关注词「{kw_str}」。"
                f"请结合上下文语境严谨甄别：\n"
                f"- 若该词确实是在直接呼唤/向机器人提问（如叫机器人出来、向机器人请教），请判定 target 为 \"bot\" 并给出 85~100 的高分；\n"
                f"- 若该词是在称呼其他群友的外号/代称、谈论他人或作为普通名词，绝不可误判为呼唤机器人，必须根据实际对象判定为 \"user\" 或 \"everyone\"，且不得给予关键词加分！\n"
            )

        prompt = (
            f"群聊记录（从旧到新）：\n{context}\n"
            f"{turn_hint}"
            f"{context_hint}"
            f"{kw_hint}\n"
            "请评估最后一条人类消息（注意：不要把机器人自己的发言当作用户消息）：\n"
            '请严格输出单行 JSON：{"target": "bot"|"user"|"everyone", "score": 数字, "reason": "分析原因"}'
        )
        raw = await self._call_text(provider, system, prompt, session_id)
        data = self._parse_json(raw) or {}
        raw_score = int(data.get("score") or 0)
        target = str(data.get("target") or "everyone").strip().lower()
        if target not in {"bot", "user", "everyone"}:
            target = "everyone"

        reason = str(data.get("reason") or "")

        if target == "bot":
            if matched_kws:
                final_score = max(raw_score, 85)
                reason = f"[关键词「{matched_kws[0]}」加权] {reason}"
            elif is_target_speaker:
                bonus = max(0, self._int("follow_score_bonus", 15))
                final_score = min(100, raw_score + bonus)
                if bonus > 0:
                    reason = f"[连续互动+{bonus}分] {reason}"
            else:
                final_score = raw_score
        else:
            final_score = raw_score

        return {
            "score": final_score,
            "raw_score": raw_score,
            "threshold": threshold,
            "target": target,
            "reason": reason,
        }

    def _judge_provider(self, session_id: str):
        provider_id = self._str("judge_provider_id", DEFAULT_JUDGE_PROVIDER).strip()
        provider = None
        if provider_id:
            try:
                provider = self.context.get_provider_by_id(provider_id)
            except Exception:
                provider = None
            if provider is None:
                logger.warning(
                    f"[GroupWatch] 指定的语境评估模型提供商 '{provider_id}' 未找到或已失效，自动回退使用当前会话的主模型。"
                )
        if provider is None:
            provider = self.context.get_using_provider(umo=session_id)
        return provider

    @staticmethod
    async def _call_text(provider, system: str, prompt: str, session_id: str) -> str:
        """调用判断模型，固定低采样温度并尝试结构化 JSON 返回。"""
        try:
            try:
                response = await asyncio.wait_for(
                    provider.text_chat(
                        prompt=prompt,
                        system_prompt=system,
                        contexts=[],
                        session_id=session_id,
                        temperature=0.1,
                        response_format={"type": "json_object"},
                    ),
                    timeout=60,
                )
            except TypeError:
                # 兼容不接收 extra kwargs 的老版本适配器
                response = await asyncio.wait_for(
                    provider.text_chat(
                        prompt=prompt,
                        system_prompt=system,
                        contexts=[],
                        session_id=session_id,
                    ),
                    timeout=60,
                )
            except Exception as call_err:
                err_str = str(call_err).lower()
                if "response_format" in err_str or "unrecognized parameter" in err_str:
                    # 模型服务商不支持 response_format，回退纯温度调用
                    response = await asyncio.wait_for(
                        provider.text_chat(
                            prompt=prompt,
                            system_prompt=system,
                            contexts=[],
                            session_id=session_id,
                            temperature=0.1,
                        ),
                        timeout=60,
                    )
                else:
                    raise call_err
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[group_watch] 判断模型调用失败: {exc}", exc_info=True)
            return ""
        return str(getattr(response, "completion_text", "") or "").strip()

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        if not raw:
            return None
        text = re.sub(r"```(?:json)?", "", raw, flags=re.I).strip()
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start : end + 1]
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except Exception:
                try:
                    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                    data = json.loads(fixed)
                    if isinstance(data, dict):
                        return data
                except Exception:
                    pass

        # 正则鲁棒兜底解析
        extracted: dict[str, Any] = {}
        target_m = re.search(r'"target"\s*:\s*"([^"]+)"', text, re.I)
        if target_m:
            extracted["target"] = target_m.group(1).strip().lower()
        score_m = re.search(r'"score"\s*:\s*(\d+)', text)
        if score_m:
            extracted["score"] = int(score_m.group(1))
        reason_m = re.search(r'"reason"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text)
        if reason_m:
            extracted["reason"] = reason_m.group(1).replace(r'\"', '"')
        return extracted if extracted else None

    # ------------------------------------------------------------------
    # 状态导出与 Web API 访问鉴权
    # ------------------------------------------------------------------
    def _status_payload(self) -> dict[str, Any]:
        now = time.time()
        groups = []
        for session_id, buf in self.state_mgr.buffers.items():
            state = self.state_mgr.get_state(
                session_id, self._bot_id, self._bot_name
            )
            follow_left = max(0, int(state["follow_until"] - now))
            silence_left = max(0, int(state.get("silence_until", 0.0) - now))
            last_human = None
            for item in reversed(buf):
                if not item.get("is_bot"):
                    h_name = str(item.get("name") or "").strip()
                    h_uid = str(item.get("uid") or "")
                    last_human = {
                        "qq": f"{h_name}({h_uid})" if h_name else h_uid,
                        "ago": max(0, int(now - item["ts"])),
                        "text": str(item.get("text") or "")[:40],
                    }
                    break
            last_bot = None
            for item in reversed(buf):
                if item.get("is_bot"):
                    last_bot = {
                        "ago": max(0, int(now - item["ts"])),
                        "text": str(item.get("text") or "")[:40],
                    }
                    break

            follow_target = ""
            if follow_left > 0:
                f_uid = str(state.get("follow_target_uid") or "")
                f_name = str(state.get("follow_target_name") or "")
                if f_uid:
                    follow_target = f"{f_name}({f_uid})" if f_name else f_uid

            groups.append(
                {
                    "umo": session_id,
                    "mode": (
                        "静默中"
                        if silence_left
                        else ("跟进中" if follow_left else "待命")
                    ),
                    "follow_left": follow_left,
                    "follow_target": follow_target,
                    "silence_left": silence_left,
                    "last_human": last_human,
                    "last_bot": last_bot,
                    "participants": self.state_mgr.get_participants(session_id),
                    "decisions": [
                        {
                            "ago": max(0, int(now - d["ts"])),
                            "time_str": datetime.fromtimestamp(d["ts"]).strftime(
                                "%H:%M:%S"
                            ),
                            "mode": d.get("mode"),
                            "related": d.get("related"),
                            "target": d.get("target", "everyone"),
                            "addressed": d.get("addressed"),
                            "reply": d.get("reply"),
                            "score": d.get("score", 0),
                            "threshold": d.get("threshold", 0),
                            "topic_state": d.get("topic_state"),
                            "reason": d.get("reason"),
                            "sent": d.get("sent"),
                            "target_text": d.get("target_text", ""),
                            "target_uid": d.get("target_uid", ""),
                            "target_name": d.get("target_name", ""),
                        }
                        for d in (state.get("decisions") or [])[-30:][::-1]
                    ],
                }
            )
        groups.sort(key=lambda g: g["umo"])
        willingness = self._int("reply_willingness", 3)
        return {
            "enabled": self._enabled,
            "bot_id": self._bot_id,
            "bot_name": self._bot_name or "机器人",
            "judge_provider_id": self._str(
                "judge_provider_id", DEFAULT_JUDGE_PROVIDER
            ),
            "judge_enabled": as_bool(self._get("judge_enabled", True), True),
            "reply_willingness": willingness,
            "reply_willingness_desc": WILLINGNESS_LABELS.get(
                willingness, f"{willingness}档"
            ),
            "follow_enabled": as_bool(self._get("follow_enabled", True), True),
            "follow_score_bonus": self._int("follow_score_bonus", 20),
            "wake_words": self._wake_words(),
            "judge_context_count": self._int("judge_context_count", 8),
            "debounce_seconds": self._int("debounce_seconds", 3),
            "repeat_trigger_count": self._int("repeat_trigger_count", 3),
            "silence_duration_minutes": self._int("silence_duration_minutes", 10),
            "follow_max_minutes": self._int("follow_max_minutes", 10),
            "groups": groups,
        }

    def _extract_request_token(self) -> str:
        """从当前请求上下文（Quart / ASGI）中提取 Authorization 或 Query token。"""
        try:
            from quart import request as q_req

            if q_req:
                auth = q_req.headers.get("Authorization", "")
                if auth:
                    if auth.lower().startswith("bearer "):
                        return auth[7:].strip()
                    return auth.strip()
                t = q_req.args.get("token", "")
                if t:
                    return str(t).strip()
        except Exception:
            pass

        try:
            from astrbot.dashboard.asgi_runtime import _request_var

            d_req = _request_var.get(None)
            if d_req:
                headers = getattr(d_req, "headers", {}) or {}
                auth = (
                    headers.get("authorization")
                    or headers.get("Authorization")
                    or ""
                )
                if auth:
                    if auth.lower().startswith("bearer "):
                        return auth[7:].strip()
                    return auth.strip()
                args = getattr(d_req, "args", {}) or {}
                t = args.get("token") or ""
                if t:
                    return str(t).strip()
        except Exception:
            pass

        return ""

    async def _api_status(self):
        token_configured = self._str("web_token", "").strip()
        if token_configured:
            token_received = self._extract_request_token()
            if token_received != token_configured:
                return (
                    {
                        "status": "error",
                        "message": "403 Forbidden: 访问被拒绝，无效或缺失看板 Token",
                    },
                    403,
                )
        return self._status_payload()

    # ------------------------------------------------------------------
    # 插件指令
    # ------------------------------------------------------------------
    @filter.command("gw")
    async def gw_command(self, event: AstrMessageEvent):
        if not event.is_admin():
            yield event.plain_result("需要管理员权限。")
            return
        raw = str(getattr(event, "message_str", "") or "")
        raw = re.sub(r"^\s*/?\s*gw\b", "", raw, flags=re.I).strip()
        arg = raw.split()[0].lower() if raw else "status"
        session_id = event.unified_msg_origin

        if arg == "status":
            yield event.plain_result(self._status_text())
        elif arg == "now":
            if session_id in self.state_mgr.buffers:
                await self._evaluate(session_id, "judge")
                state = self.state_mgr.get_state(
                    session_id, self._bot_id, self._bot_name
                )
                last = (state.get("decisions") or [None])[-1]
                if last:
                    score = last.get("score", 0)
                    threshold = last.get("threshold", 0)
                    rel_s = (
                        f"达标({score}/{threshold}分)"
                        if last.get("related")
                        else f"未达标({score}/{threshold}分)"
                    )
                    rep_s = (
                        "已唤醒接话"
                        if last.get("sent")
                        else ("决定接话" if last.get("reply") else "不接话")
                    )
                    yield event.plain_result(
                        f"即时判定完成：\n"
                        f"• 结论：[{rel_s} | {rep_s}]\n"
                        f"• 发言对象：{last.get('target', 'everyone')}\n"
                        f"• 理由：{last.get('reason') or '无'}"
                    )
                else:
                    yield event.plain_result("评估完成。")
            else:
                yield event.plain_result("这个群还没有缓存消息。")
        elif arg == "on":
            self._enabled = True
            yield event.plain_result("群聊巡视已开启（仅本次运行）。")
        elif arg == "off":
            self._enabled = False
            yield event.plain_result("群聊巡视已暂停（仅本次运行）。")
        elif arg == "silence":
            parts = raw.split()
            s_min = max(1, self._int("silence_duration_minutes", 10))
            if len(parts) >= 2:
                try:
                    s_min = int(parts[1])
                except ValueError:
                    pass
            now = time.time()
            if s_min <= 0:
                self.state_mgr.clear_silence(session_id)
                yield event.plain_result("已解除当前群聊的静默状态。")
            else:
                self.state_mgr.set_silence(session_id, now + s_min * 60)
                state = self.state_mgr.get_state(session_id, self._bot_id, self._bot_name)
                state["follow_until"] = 0.0
                state["follow_from"] = 0.0
                state["follow_target_uid"] = ""
                state["follow_target_name"] = ""
                state["unrelated"] = 0
                yield event.plain_result(f"当前群聊已进入静默期 {s_min} 分钟（可随时发送“/gw silence 0”提前解除）。")
        else:
            yield event.plain_result("用法：/gw status | now | on | off | silence [分钟数]")

    def _status_text(self) -> str:
        if not self.state_mgr.buffers:
            return "还没有收到群消息。"
        now = time.time()
        lines = ["【群聊巡视状态】"]
        for session_id, buf in self.state_mgr.buffers.items():
            state = self.state_mgr.get_state(
                session_id, self._bot_id, self._bot_name
            )
            follow = max(0, int(state["follow_until"] - now) // 60)
            silence = max(0, int(state.get("silence_until", 0.0) - now))
            decisions = state.get("decisions") or []
            last = decisions[-1] if decisions else None
            reason = last.get("reason") if last else "暂无"
            score = last.get("score", 0) if last else 0
            threshold = last.get("threshold", 0) if last else 0
            target = last.get("target", "everyone") if last else "everyone"
            rel_s = (
                f"达标({score}/{threshold}分)"
                if (last and last.get("related"))
                else f"未达标({score}/{threshold}分)"
            )
            sent_s = "已接话" if (last and last.get("sent")) else "未接话"
            mode_s = (
                "静默中"
                if silence > 0
                else ("跟进中" if follow > 0 else "待命巡视")
            )
            status_desc = (
                f"静默倒计时: {silence // 60}分{silence % 60}秒"
                if silence > 0
                else f"跟进倒计时: {follow} 分钟"
            )
            lines.append(
                f"- 群聊：{session_id} [{mode_s}]\n"
                f"  缓存消息: {len(buf)} 条 | {status_desc}\n"
                f"  最近判定: [{rel_s} | {sent_s} | 对象:{target}] 理由: {reason}"
            )
        willingness = self._int("reply_willingness", 3)
        lines.append(
            f"• 总开关: {'开' if self._enabled else '关'} | 积极度: {willingness}档 | 机器人: {self._bot_name}({self._bot_id})\n"
            f"• 判定模型: {self._str('judge_provider_id', DEFAULT_JUDGE_PROVIDER)}\n"
            f"💡 提示：在 WebUI 插件页面打开【群聊巡视状态】看板可实时监控所有判定！"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        self._enabled = bool(self._get("enabled", True))
        self._sync_judge_options()

        try:
            platforms = self.context.platform_manager.get_insts() or []
            for p in platforms:
                sid = getattr(p, "self_id", None) or getattr(p, "bot_id", None)
                if sid:
                    self._bot_id = str(sid)
                sname = getattr(p, "self_name", None) or getattr(p, "bot_name", None)
                if sname:
                    self._bot_name = str(sname)
                if self._bot_id:
                    break
        except Exception:
            pass

        for route in ("/astrbot_plugin_group_watch/status", "/group_watch/status"):
            try:
                self.context.register_web_api(
                    route, self._api_status, ["GET"], "群聊巡视状态"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[group_watch] 注册状态接口失败: {exc}")
        logger.info(
            "[group_watch] 已启动：判断模型 %s，上下文 %s 条，积极度 %s 档，跟进最长 %s 分钟",
            self._str("judge_provider_id", DEFAULT_JUDGE_PROVIDER),
            self._int("judge_context_count", 8),
            self._int("reply_willingness", 3),
            self._int("follow_max_minutes", 10),
        )

    async def terminate(self) -> None:
        self.img_service.cancel_all()
        self.state_mgr.cancel_all_debounce()
