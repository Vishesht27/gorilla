from bfcl_eval.model_handler.local_inference.qwen import QwenHandler
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from overrides import override

# Qwen3 chat template disables thinking when generation starts with an empty block:
# <|im_start|>assistant\n<think>\n\n</think>\n\n
_NO_THINK_SUFFIX = "<think>\n\n</think>\n\n"


class QwenNoThinkHandler(QwenHandler):
    """Local Qwen3 prompt handler with thinking/reasoning disabled."""

    @override
    def _format_prompt(self, messages, function):
        return super()._format_prompt(messages, function) + _NO_THINK_SUFFIX


class QwenFCNoThinkHandler(QwenFCHandler):
    """Local Qwen3 FC handler with thinking/reasoning disabled."""

    @override
    def _format_prompt(self, messages, function):
        return super()._format_prompt(messages, function) + _NO_THINK_SUFFIX
