import json

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override


class LummaHandler(OSSHandler):
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

    @override
    def _format_prompt(self, messages, function):
        # Lumma's chat template consumes tools via the custom `tools_list=` kwarg
        return self.tokenizer.apply_chat_template(
            messages,
            tools_list=json.dumps(function),
            add_generation_prompt=True,
            tokenize=False,
        )

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        # Do NOT inject BFCL's default system prompt; the chat template handles tools.
        functions = test_entry["function"]
        return {"message": [], "function": functions}

    # decode_ast / decode_execute are inherited from OSSHandler and already handle
    # the Pythonic `[FuncName(param=value)]` format Lumma emits.
