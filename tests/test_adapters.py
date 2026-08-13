# Fugal — Apache-2.0. See NOTICE.
"""Unit tests for the pure functions in serve.py — the wire-shape adapters.

    python -m unittest discover -s tests -v

No network, no API key, no backbone, no spend: every function under test is a plain
data transform. That is exactly why they are worth testing here rather than by hand
against a live client — these are the places where a bug does not raise, it just
produces a response an agent harness quietly mishandles. The comments in serve.py
name the two that already bit: a dropped `tool_call_id` breaks a tool loop with no
error, and a system prompt faked as a user turn reads to the worker as conversation.

Importing fugal.serve pulls in numpy (via router.py) but NOT torch, transformers or
requests — those are imported lazily at the point of use.
"""
import json
import unittest

from fugal.serve import (anthropic_response, anthropic_text, anthropic_to_body,
                         anthropic_tools_to_openai, clean_history,
                         client_system_from_body, openai_calls_to_anthropic)
from fugal.router import WORKER_SYSTEM_PROMPT, clamp_max_tokens, compose_system


class TestClampMaxTokens(unittest.TestCase):
    def test_passthrough(self):
        self.assertEqual(clamp_max_tokens(512), 512)
        self.assertEqual(clamp_max_tokens("512"), 512)      # OpenAI clients send strings

    def test_junk_and_absent_fall_back(self):
        for bad in (None, "", "abc", {}, [], 3.5j):
            self.assertEqual(clamp_max_tokens(bad), 4096, bad)

    def test_nonpositive_falls_back_rather_than_clamping_to_one(self):
        # 0 means "unset" in most clients; honouring it literally would return an
        # empty completion that looks like the model refusing.
        self.assertEqual(clamp_max_tokens(0), 4096)
        self.assertEqual(clamp_max_tokens(-5), 4096)

    def test_upper_bound(self):
        self.assertEqual(clamp_max_tokens(10 ** 9), 32000)


class TestComposeSystem(unittest.TestCase):
    def test_house_prompt_alone(self):
        for empty in (None, "", "   \n "):
            self.assertEqual(compose_system(empty), WORKER_SYSTEM_PROMPT)

    def test_client_prompt_is_appended_never_dropped(self):
        out = compose_system("You are a git expert.")
        self.assertTrue(out.startswith(WORKER_SYSTEM_PROMPT))
        self.assertTrue(out.endswith("You are a git expert."))

    def test_house_prompt_leads(self):
        # Order is load-bearing: the house prompt pins one identity across models, so it
        # must not end up after instructions that could contradict it.
        out = compose_system("Say you are GPT-4.")
        self.assertLess(out.index(WORKER_SYSTEM_PROMPT), out.index("Say you are GPT-4."))


class TestClientSystemFromBody(unittest.TestCase):
    def test_anthropic_top_level_field(self):
        self.assertEqual(client_system_from_body({"system": "be terse"}), "be terse")

    def test_openai_system_messages_are_joined(self):
        body = {"messages": [{"role": "system", "content": "a"},
                             {"role": "user", "content": "q"},
                             {"role": "system", "content": "b"}]}
        self.assertEqual(client_system_from_body(body), "a\n\nb")

    def test_absent(self):
        self.assertEqual(client_system_from_body({"messages": []}), "")
        self.assertEqual(client_system_from_body({}), "")

    def test_non_string_system_content_is_ignored(self):
        body = {"messages": [{"role": "system", "content": [{"type": "text", "text": "x"}]}]}
        self.assertEqual(client_system_from_body(body), "")


class TestAnthropicText(unittest.TestCase):
    def test_bare_string(self):
        self.assertEqual(anthropic_text("hello"), "hello")

    def test_text_blocks_joined(self):
        self.assertEqual(anthropic_text([{"type": "text", "text": "a"},
                                         {"type": "text", "text": "b"}]), "a\nb")

    def test_non_text_blocks_are_dropped_not_stringified(self):
        # An image placeholder in the prompt is worse than its absence — the router
        # would route on the word "image" rather than on the question.
        out = anthropic_text([{"type": "image", "source": {"data": "..."}},
                              {"type": "text", "text": "what is this?"}])
        self.assertEqual(out, "what is this?")

    def test_unknown_shapes(self):
        self.assertEqual(anthropic_text(None), "")
        self.assertEqual(anthropic_text(42), "")


class TestToolDefinitionConversion(unittest.TestCase):
    def test_shape(self):
        out = anthropic_tools_to_openai([{"name": "read_file", "description": "reads",
                                          "input_schema": {"type": "object",
                                                           "properties": {"p": {}}}}])
        self.assertEqual(out, [{"type": "function", "function": {
            "name": "read_file", "description": "reads",
            "parameters": {"type": "object", "properties": {"p": {}}}}}])

    def test_missing_fields_get_valid_defaults(self):
        out = anthropic_tools_to_openai([{"name": "t"}])
        self.assertEqual(out[0]["function"]["description"], "")
        # A tool with no parameters schema must still present a valid JSON Schema object,
        # or the provider rejects the whole request.
        self.assertEqual(out[0]["function"]["parameters"],
                         {"type": "object", "properties": {}})

    def test_nameless_and_malformed_entries_are_skipped(self):
        self.assertEqual(anthropic_tools_to_openai([{"description": "no name"}, "junk", None]),
                         [])

    def test_empty(self):
        self.assertEqual(anthropic_tools_to_openai(None), [])


class TestToolCallConversion(unittest.TestCase):
    def test_arguments_string_becomes_object(self):
        out = openai_calls_to_anthropic([{"id": "call_1", "function": {
            "name": "grep", "arguments": '{"pattern": "TODO"}'}}])
        self.assertEqual(out, [{"type": "tool_use", "id": "call_1", "name": "grep",
                                "input": {"pattern": "TODO"}}])

    def test_malformed_json_keeps_the_block(self):
        # Dropping it would hang the client forever waiting for a tool call that never
        # arrives; an empty input lets it reply with an error instead.
        out = openai_calls_to_anthropic([{"id": "c", "function": {
            "name": "f", "arguments": "{not json"}}])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["input"], {})

    def test_non_object_json_arguments(self):
        out = openai_calls_to_anthropic([{"id": "c", "function": {
            "name": "f", "arguments": "[1, 2]"}}])
        self.assertEqual(out[0]["input"], {})

    def test_missing_id_is_synthesised(self):
        out = openai_calls_to_anthropic([{"function": {"name": "f", "arguments": "{}"}}])
        self.assertTrue(out[0]["id"].startswith("toolu_"))

    def test_empty(self):
        self.assertEqual(openai_calls_to_anthropic(None), [])


class TestAnthropicToBody(unittest.TestCase):
    def test_system_is_lifted_not_faked_as_a_user_turn(self):
        body = anthropic_to_body({"system": "be terse",
                                  "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(body["system"], "be terse")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])

    def test_system_blocks_are_flattened(self):
        body = anthropic_to_body({"system": [{"type": "text", "text": "be terse"}],
                                  "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(body["system"], "be terse")

    def test_no_system_key_when_absent(self):
        self.assertNotIn("system", anthropic_to_body(
            {"messages": [{"role": "user", "content": "hi"}]}))

    def test_content_blocks_flatten_to_strings(self):
        body = anthropic_to_body({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}]})
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])

    def test_tool_use_becomes_openai_tool_calls(self):
        body = anthropic_to_body({"messages": [
            {"role": "assistant", "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "toolu_1", "name": "ls", "input": {"path": "/"}}]}]})
        msg = body["messages"][0]
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "checking")
        self.assertEqual(msg["tool_calls"][0]["id"], "toolu_1")
        self.assertEqual(msg["tool_calls"][0]["type"], "function")
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "ls")
        # arguments is a JSON *string* on the OpenAI side, not an object
        self.assertEqual(json.loads(msg["tool_calls"][0]["function"]["arguments"]),
                         {"path": "/"})

    def test_tool_result_carries_tool_call_id(self):
        # THE regression. Without tool_call_id a model cannot match a result to the call
        # it made, and the tool loop cannot proceed — with no error anywhere.
        body = anthropic_to_body({"messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "bin  etc"}]}]})
        self.assertEqual(body["messages"], [
            {"role": "tool", "tool_call_id": "toolu_1", "content": "bin  etc"}])

    def test_full_tool_round_trip_preserves_id_name_and_input(self):
        original = {"type": "tool_use", "id": "toolu_abc", "name": "grep",
                    "input": {"pattern": "x", "n": 3}}
        body = anthropic_to_body({"messages": [
            {"role": "assistant", "content": [original]}]})
        back = openai_calls_to_anthropic(body["messages"][0]["tool_calls"])
        self.assertEqual(back, [original])

    def test_assistant_with_only_tool_use_has_null_content(self):
        body = anthropic_to_body({"messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t", "name": "f", "input": {}}]}]})
        self.assertIsNone(body["messages"][0]["content"])

    def test_unknown_roles_and_empty_turns_are_dropped(self):
        body = anthropic_to_body({"messages": [
            {"role": "system", "content": "ignored here"},
            {"role": "user", "content": "   "},
            "junk",
            {"role": "user", "content": "real"}]})
        self.assertEqual(body["messages"], [{"role": "user", "content": "real"}])

    def test_empty_request(self):
        self.assertEqual(anthropic_to_body({}), {"messages": []})


class TestAnthropicResponse(unittest.TestCase):
    def test_reports_the_model_that_actually_answered(self):
        r = anthropic_response("req_abc", "hi", {"final_model": "openai/gpt-5.5"})
        self.assertEqual(r["model"], "fugal/openai/gpt-5.5")
        self.assertEqual(r["id"], "msg_abc")

    def test_end_turn_vs_tool_use(self):
        self.assertEqual(anthropic_response("req_1", "hi", {})["stop_reason"], "end_turn")
        r = anthropic_response("req_1", "", {"tool_calls": [
            {"id": "c", "function": {"name": "f", "arguments": "{}"}}]})
        self.assertEqual(r["stop_reason"], "tool_use")
        self.assertEqual([b["type"] for b in r["content"]], ["tool_use"])

    def test_text_precedes_tool_blocks(self):
        r = anthropic_response("req_1", "thinking", {"tool_calls": [
            {"id": "c", "function": {"name": "f", "arguments": "{}"}}]})
        self.assertEqual([b["type"] for b in r["content"]], ["text", "tool_use"])

    def test_never_returns_empty_content(self):
        # A zero-block content array is not valid in the Messages shape; clients differ
        # on whether they raise or hang.
        r = anthropic_response("req_1", "", {})
        self.assertEqual(r["content"], [{"type": "text", "text": ""}])

    def test_usage_block_is_real_integers(self):
        r = anthropic_response("req_1", "hi", {"input_tokens": 12, "output_tokens": 34})
        self.assertEqual(r["usage"], {"input_tokens": 12, "output_tokens": 34})
        self.assertEqual(anthropic_response("req_1", "hi", {})["usage"],
                         {"input_tokens": 0, "output_tokens": 0})


class TestCleanHistory(unittest.TestCase):
    def _body(self, n):
        return {"messages": [{"role": "user" if i % 2 == 0 else "assistant",
                              "content": f"m{i}"} for i in range(n)]}

    def test_last_message_is_excluded(self):
        # The last message is the query being routed; including it would send it twice.
        hist = clean_history(self._body(3))
        self.assertEqual([m["content"] for m in hist], ["m0", "m1"])

    def test_message_count_cap_drops_oldest(self):
        hist = clean_history(self._body(30), max_msgs=4)
        self.assertEqual(len(hist), 4)
        self.assertEqual(hist[-1]["content"], "m28")

    def test_char_budget_drops_oldest(self):
        body = {"messages": [{"role": "user", "content": "x" * 5000} for _ in range(6)]}
        hist = clean_history(body, max_msgs=12, max_chars=12000)
        self.assertLessEqual(sum(len(m["content"]) for m in hist), 12000)

    def test_per_message_truncation(self):
        body = {"messages": [{"role": "user", "content": "x" * 20000},
                             {"role": "user", "content": "q"}]}
        self.assertEqual(len(clean_history(body)[0]["content"]), 8000)

    def test_non_conversational_turns_are_dropped(self):
        body = {"messages": [{"role": "system", "content": "s"},
                             {"role": "tool", "tool_call_id": "t", "content": "r"},
                             {"role": "user", "content": ""},
                             {"role": "user", "content": None},
                             "junk",
                             {"role": "assistant", "content": "keep"},
                             {"role": "user", "content": "q"}]}
        self.assertEqual(clean_history(body), [{"role": "assistant", "content": "keep"}])

    def test_empty(self):
        self.assertEqual(clean_history({}), [])
        self.assertEqual(clean_history({"messages": []}), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
