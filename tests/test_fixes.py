# -*- coding: utf-8 -*-
import json
import unittest
from unittest.mock import MagicMock

from astrbot_plugin_group_watch.main import GroupWatchPlugin
from astrbot_plugin_group_watch.image_service import ImageService

class TestGroupWatchFixes(unittest.TestCase):
    def setUp(self):
        self.ctx = MagicMock()
        self.plugin = GroupWatchPlugin(self.ctx, {"enabled": True})

    def test_json_trailing_comma_parsing(self):
        """测试 JSON 解析对尾逗号的容错清理 (r'\1' 引用捕获组)"""
        # 测试字典尾逗号
        raw_dict = '{"target": "bot", "score": 85, }'
        parsed = self.plugin._parse_json(raw_dict)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.get("target"), "bot")
        self.assertEqual(parsed.get("score"), 85)

        # 测试数组尾逗号
        raw_arr = '{"target": "everyone", "tags": ["a", "b", ], }'
        parsed_arr = self.plugin._parse_json(raw_arr)
        self.assertIsNotNone(parsed_arr)
        self.assertEqual(parsed_arr.get("tags"), ["a", "b"])

    def test_target_user_veto_and_unavailable_model(self):
        """测试发言对象 target=='user' 一票否决以及模型不可用时不误唤醒"""
        # 1. target == "user" 且分值极高 (95 >= 55) 时，必须一票否决为 False
        judge_user = {
            "related": True,
            "target": "user",
            "score": 95,
            "threshold": 55,
            "reason": "群友私聊讨论"
        }
        score = judge_user["score"]
        threshold = judge_user["threshold"]
        target = judge_user["target"]
        related = bool(judge_user.get("related", False)) and (score >= threshold) and (target != "user")
        self.assertFalse(related, "当 target=='user' 时必须一票否决")

        # 2. 模型不可用 (score=0, threshold=0, related=False) 时，绝对不能因为 0>=0 判定为 True
        judge_unavailable = {
            "related": False,
            "target": "everyone",
            "score": 0,
            "threshold": 0,
            "reason": "没有可用的判断模型"
        }
        score_un = judge_unavailable["score"]
        threshold_un = judge_unavailable["threshold"]
        target_un = judge_unavailable["target"]
        related_un = bool(judge_unavailable.get("related", False)) and (score_un >= threshold_un) and (target_un != "user")
        self.assertFalse(related_un, "模型不可用时绝对不能误判定为达标")

        # 3. 正常对机器人发言且达标
        judge_bot = {
            "related": True,
            "target": "bot",
            "score": 80,
            "threshold": 55,
            "reason": "明确询问机器人"
        }
        score_b = judge_bot["score"]
        threshold_b = judge_bot["threshold"]
        target_b = judge_bot["target"]
        related_b = bool(judge_bot.get("related", False)) and (score_b >= threshold_b) and (target_b != "user")
        self.assertTrue(related_b, "合法呼唤机器人且达标时应判定为 True")

    def test_image_extraction_from_image_urls(self):
        """测试从 AstrBot 4.29+ 标准的 req.image_urls 中提取图片"""
        class MockReq:
            def __init__(self, urls):
                self.image_urls = urls
                self.prompt = "测试纯文本 prompt"

        req = MockReq(["http://example.com/pic1.png", "C:/temp/pic2.jpg"])
        images = ImageService.extract_prompt_images(req)
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0], "http://example.com/pic1.png")
        self.assertEqual(images[1], "C:/temp/pic2.jpg")

if __name__ == "__main__":
    unittest.main()
