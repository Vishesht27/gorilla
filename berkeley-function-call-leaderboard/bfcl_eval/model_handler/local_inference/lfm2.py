from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import (
    default_decode_ast_prompting,
    default_decode_execute_prompting,
)
from overrides import override


class LFM2Handler(OSSHandler):
    """Handler for LiquidAI LFM2 models.

    LFM2 uses a native tool paradigm very similar to Lumma: tool defs go in the
    system prompt between <|tool_list_start|>/<|tool_list_end|>, and the model
    emits Pythonic calls between <|tool_call_start|>/<|tool_call_end|>, e.g.
    `[get_status(candidate_id="12345")]`.
    """

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        # Keep special tokens in the response so we can locate the tool-call section.
        self.skip_special_tokens = False

    @override
    def _format_prompt(self, messages, function):
        # LFM2's chat template consumes tools natively via the `tools=` kwarg.
        return self.tokenizer.apply_chat_template(
            messages,
            tools=function,
            add_generation_prompt=True,
            tokenize=False,
        )

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        # LFM2 handles tools via its own template; don't inject BFCL's system prompt.
        functions = test_entry["function"]
        return {"message": [], "function": functions}

    @staticmethod
    def _extract_tool_calls(result: str) -> str:
        if "<|tool_call_start|>" in result:
            result = result.split("<|tool_call_start|>", 1)[1]
            result = result.split("<|tool_call_end|>", 1)[0]
        # Strip any remaining special tokens / plain-text trailer.
        for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
            result = result.replace(tok, "")
        return result.strip()

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        return default_decode_ast_prompting(
            self._extract_tool_calls(result), language, has_tool_call_tag
        )

    @override
    def decode_execute(self, result, has_tool_call_tag):
        return default_decode_execute_prompting(
            self._extract_tool_calls(result), has_tool_call_tag
        )
