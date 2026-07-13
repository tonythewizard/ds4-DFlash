#!/usr/bin/env python3
"""Unit tests for the DS4 terminal client; no running model is required."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/ds4_ui.py"
SPEC = importlib.util.spec_from_file_location("ds4_ui", MODULE_PATH)
assert SPEC and SPEC.loader
ui = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ui
SPEC.loader.exec_module(ui)


class FakeWindow:
    def __init__(self, height: int = 30, width: int = 120) -> None:
        self.height = height
        self.width = width
        self.output: list[str] = []

    def getmaxyx(self):
        return self.height, self.width

    def addnstr(self, _y, _x, text, length, _attr=0):
        self.output.append(str(text)[:length])

    def erase(self):
        self.output.clear()

    def refresh(self):
        pass


def args(**overrides):
    values = dict(
        server="http://127.0.0.1:8080",
        proxy="http://127.0.0.1:8081",
        sidecar_health="http://127.0.0.1:8091/health",
        sidecar_generate="http://127.0.0.1:8091/propose",
        dflash="off",
        language="en",
        max_tokens=32,
        prefill_tokens=2,
        temperature=0.0,
        top_k=8,
        seed=1,
        timeout=30.0,
        stream=False,
        system="Be clear, useful, and direct.",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class PrefillMetricTests(unittest.TestCase):
    def test_positive_metric(self):
        self.assertEqual(ui.format_prefill_metric(22, 110), "PREFILL 22 tok 200.00 tok/s")

    def test_unusable_times_are_na(self):
        for value in (None, 0, -1, float("nan"), float("inf"), "bad"):
            with self.subTest(value=value):
                self.assertEqual(ui.format_prefill_metric(22, value), "PREFILL 22 tok N/A")

    def test_numeric_strings(self):
        self.assertEqual(ui.format_prefill_metric("10", "250"), "PREFILL 10 tok 40.00 tok/s")

    def test_off_on_and_fallback_shapes_share_normalizer(self):
        responses = {
            "off": {"usage": {"prompt_tokens": 20}, "prompt_processing_ms": 100},
            "on": {"usage": {"prompt_tokens": 20}, "ds4_runtime": {"prompt_processing_ms": 100}},
            "fallback": {"ds4_runtime": {"prompt_token_count": "20", "prompt_processing_ms": "100"}},
        }
        displays = []
        for response in responses.values():
            displays.append(ui.format_prefill_metric(*ui.prompt_metrics(response)))
        self.assertEqual(displays, ["PREFILL 20 tok 200.00 tok/s"] * 3)

    def test_missing_backend_timing_is_not_wall_clock(self):
        tokens, milliseconds = ui.prompt_metrics({"usage": {"prompt_tokens": 22}})
        self.assertEqual(tokens, 22)
        self.assertIsNone(milliseconds)
        self.assertEqual(ui.TurnStats(prompt_tokens=tokens).prefill_display, "PREFILL 22 tok N/A")


class ThinkingParserTests(unittest.TestCase):
    def parse_chunks(self, *chunks):
        parser = ui.ThinkingParser()
        for chunk in chunks:
            parser.feed(chunk)
        return parser.finish()

    def test_complete_and_arbitrarily_split_tags(self):
        source = "<think>private reasoning</think>public answer"
        expected = ui.AssistantContent("private reasoning", "public answer")
        self.assertEqual(self.parse_chunks(source), expected)
        parser = ui.ThinkingParser()
        for character in source:
            parser.feed(character)
        self.assertEqual(parser.finish(), expected)

    def test_stream_snapshot_hides_partial_tag_and_updates_thinking(self):
        parser = ui.ThinkingParser()
        parser.feed("<think>visible reasoning</thi")
        snapshot = parser.snapshot()
        self.assertTrue("visible reasoning".startswith(snapshot.thinking))
        self.assertGreater(len(snapshot.thinking), 0)
        self.assertEqual(snapshot.final, "")
        self.assertNotIn("</thi", snapshot.thinking + snapshot.final)
        parser.feed("nk>final")
        self.assertEqual(parser.snapshot().final, "")
        self.assertEqual(parser.finish().final, "final")

    def test_stream_metadata_merge(self):
        merged = ui.merge_stream_metadata([
            {"choices": [{"delta": {"content": "a"}}]},
            {"usage": {"prompt_tokens": 9, "completion_tokens": 1},
             "ds4_runtime": {"prompt_processing_ms": 30}},
        ])
        self.assertEqual(ui.prompt_metrics(merged), (9, 30.0))

    def test_stray_close_and_unclosed_open(self):
        self.assertEqual(
            self.parse_chunks("legacy reasoning</thi", "nk>answer"),
            ui.AssistantContent("legacy reasoning", "answer"),
        )
        self.assertEqual(
            self.parse_chunks("<think>unfinished"),
            ui.AssistantContent("unfinished", ""),
        )

    def test_opening_and_closing_tags_split_independently(self):
        self.assertEqual(
            self.parse_chunks("<thi", "nk>reason", "</thi", "nk>final"),
            ui.AssistantContent("reason", "final"),
        )

    def test_final_in_same_chunk_and_empty_reasoning(self):
        self.assertEqual(
            self.parse_chunks("<think></think>final"),
            ui.AssistantContent("", "final"),
        )

    def test_no_thinking_and_empty(self):
        self.assertEqual(self.parse_chunks("answer only"), ui.AssistantContent("", "answer only"))
        self.assertEqual(self.parse_chunks(""), ui.AssistantContent())

    def test_multiple_blocks_do_not_leak_tags(self):
        result = self.parse_chunks("<think>a</think>A<think>b</think>B")
        self.assertEqual(result, ui.AssistantContent("ab", "AB"))
        self.assertNotIn("think", result.final)

    def test_structured_reasoning_has_priority(self):
        result = ui.parse_assistant_responses([{
            "choices": [{"message": {
                "reasoning_content": "structured",
                "content": "<think>legacy</think>final",
            }}]
        }])
        self.assertEqual(result, ui.AssistantContent("structured", "final"))

    def test_structured_thinking_and_duplicate_aliases(self):
        result = ui.parse_assistant_responses([{
            "reasoning_content": "one copy",
            "thinking": "one copy",
            "content": "final",
        }])
        self.assertEqual(result, ui.AssistantContent("one copy", "final"))

    def test_fixed_dflash_schema_routes_captured_reasoning(self):
        fixture = json.loads(
            (MODULE_PATH.parents[1] / "tests/fixtures/dflash_reasoning_truncation.json")
            .read_text(encoding="utf-8")
        )
        result = ui.parse_assistant_responses([fixture["prefill"], fixture["dflash"]])
        self.assertEqual(result.__dict__, fixture["expected"])
        self.assertNotIn("Abbiamo", result.final)

    def test_dflash_on_and_off_identical_final_stream(self):
        off = [{"choices": [{"message": {"content": "<think>r</think>same final"}}]}]
        on = [
            {"choices": [{"message": {"content": "<thi"}}]},
            {"text": "nk>r</think>same final"},
        ]
        off_result = ui.parse_assistant_responses(off)
        on_result = ui.parse_assistant_responses(on)
        self.assertEqual(off_result, on_result)
        self.assertEqual(on_result.final, "same final")

    def test_streaming_delta_shape_and_unicode(self):
        responses = [
            {"choices": [{"delta": {"reasoning": "ragionamento 🧠"}}]},
            {"choices": [{"delta": {"content": "risposta è pronta"}}]},
        ]
        self.assertEqual(
            ui.parse_assistant_responses(responses),
            ui.AssistantContent("ragionamento 🧠", "risposta è pronta"),
        )

    def test_thinking_box_empty_localized_and_ansi_removed(self):
        english = "\n".join(ui.thinking_box_lines("", "en", 50))
        italian = "\n".join(ui.thinking_box_lines("\x1b[31msegreto\x1b[0m", "it", 50))
        self.assertIn("THINKING", english)
        self.assertIn("No thinking stream received", english)
        self.assertIn("RAGIONAMENTO", italian)
        self.assertIn("segreto", italian)
        self.assertNotIn("\x1b", italian)


class LanguageTests(unittest.TestCase):
    def make_tui(self, **overrides):
        return ui.DS4TUI(FakeWindow(), args(**overrides))

    def test_aliases_case_and_invalid(self):
        for value, expected in (("EN", "en"), ("English", "en"), ("IT", "it"),
                                ("Italian", "it"), ("ITALIANO", "it"), ("fr", None)):
            self.assertEqual(ui.normalize_language(value), expected)

    def test_default_argument_is_english(self):
        original = sys.argv
        try:
            sys.argv = [str(MODULE_PATH)]
            self.assertEqual(ui.parse_args().language, "en")
        finally:
            sys.argv = original

    def test_language_command_and_history_preservation(self):
        tui = self.make_tui()
        tui.messages.extend([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])
        before = list(tui.messages)
        tui.handle_command("/language ITALIANO")
        self.assertEqual(tui.language, "it")
        self.assertEqual(tui.messages, before)
        self.assertIn("Lingua impostata", tui.transcript[-1][1])
        tui.handle_command("/language")
        self.assertIn("IT", tui.transcript[-1][1])
        tui.handle_command("/language klingon")
        self.assertIn("non supportata", tui.transcript[-1][1])
        tui.handle_command("/language english")
        self.assertEqual(tui.language, "en")
        self.assertIn("English", tui.transcript[-1][1])

    def test_exactly_one_canonical_instruction_and_switch(self):
        history = [
            {"role": "system", "content": "base"},
            {"role": "system", "content": ui.LANGUAGE_INSTRUCTIONS["en"]},
            {"role": "user", "content": "keep me"},
            {"role": "system", "content": ui.LANGUAGE_INSTRUCTIONS["it"]},
        ]
        original = [dict(item) for item in history]
        outgoing = ui.build_request_messages(history, "it")
        language_messages = [m for m in outgoing if m["content"] in ui.LANGUAGE_INSTRUCTIONS.values()]
        self.assertEqual(language_messages, [{"role": "system", "content": ui.LANGUAGE_INSTRUCTIONS["it"]}])
        self.assertIn({"role": "user", "content": "keep me"}, outgoing)
        self.assertEqual(history, original)

    def test_header_marker_help_reset_and_always_visible_box(self):
        tui = self.make_tui()
        tui.health = {"server": True, "sidecar": True}
        tui._last_health_poll = float("inf")
        tui.handle_command("/help")
        self.assertIn("/language en", tui.transcript[-1][1])
        tui.add_thinking("")
        wrapped = "\n".join(line for _, line in tui.wrapped_lines(80))
        self.assertIn("THINKING", wrapped)
        self.assertIn("No thinking stream received", wrapped)
        tui.draw()
        self.assertTrue(any("LANG:EN" in line for line in tui.stdscr.output))
        tui.handle_command("/reset")
        self.assertFalse(any(role.startswith("THINKING:") for role, _ in tui.transcript))
        self.assertIsNone(tui.live_content)

    def test_payload_explicitly_disables_stochastic_thinking_and_has_one_language_instruction(self):
        tui = self.make_tui(language="it")
        payload = tui.chat_payload(tui.messages + [{"role": "user", "content": "ciao"}])
        self.assertIs(payload["thinking"], False)
        language_messages = [
            message for message in payload["messages"]
            if message["content"] in ui.LANGUAGE_INSTRUCTIONS.values()
        ]
        self.assertEqual(language_messages, [
            {"role": "system", "content": ui.LANGUAGE_INSTRUCTIONS["it"]}
        ])
        self.assertIn("ragionamento visibile", language_messages[0]["content"])


class TokenLimitTests(unittest.TestCase):
    def make_tui(self, language="en"):
        return ui.DS4TUI(FakeWindow(), args(language=language))

    def test_query_and_valid_update_are_future_only(self):
        tui = self.make_tui()
        current_payload = tui.chat_payload(tui.messages)
        history = list(tui.messages)
        tui.handle_command("/max-tokens")
        self.assertIn("32", tui.transcript[-1][1])
        tui.handle_command("/max-tokens 64")
        self.assertEqual(tui.args.max_tokens, 64)
        self.assertEqual(tui.messages, history)
        self.assertEqual(current_payload["max_tokens"], 32)
        self.assertEqual(tui.chat_payload(tui.messages)["max_tokens"], 64)

    def test_invalid_text_below_and_above_range(self):
        tui = self.make_tui()
        for command in ("/max-tokens nope", "/max-tokens 2", "/max-tokens 257"):
            with self.subTest(command=command):
                tui.handle_command(command)
                self.assertEqual(tui.args.max_tokens, 32)
                self.assertIn("Invalid", tui.transcript[-1][1])

    def test_localized_limit_notice_and_finish_detection(self):
        tui = self.make_tui("it")
        tui.events.put(("answer", {
            "prompt": "x", "thinking": "ragionamento", "answer": "risposta",
            "messages": tui.messages + [{"role": "user", "content": "x"}],
            "stats": ui.asdict(ui.TurnStats(
                finish_reason="length", stop_reason="max_tokens",
                requested_max_tokens=32,
            )),
        }))
        tui.process_events()
        self.assertEqual(tui.transcript[-1][0], "NOTICE")
        self.assertEqual(
            tui.transcript[-1][1],
            "[Risposta interrotta al limite di token configurato.]",
        )
        self.assertTrue(ui.token_limit_reached("length", ""))
        self.assertFalse(ui.token_limit_reached("stop", "eos"))

    def test_argument_range_and_default(self):
        self.assertEqual(ui.max_tokens_argument("3"), 3)
        self.assertEqual(ui.max_tokens_argument("256"), 256)
        for value in ("bad", "2", "257"):
            with self.assertRaises(argparse.ArgumentTypeError):
                ui.max_tokens_argument(value)
        original = sys.argv
        try:
            sys.argv = [str(MODULE_PATH)]
            self.assertEqual(ui.parse_args().max_tokens, 256)
        finally:
            sys.argv = original


if __name__ == "__main__":
    unittest.main()
