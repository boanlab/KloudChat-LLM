"""Contract check executed inside the pinned LiteLLM proxy image in CI."""

from __future__ import annotations

import datetime
import json
import os

import litellm
from litellm.proxy.proxy_server import general_settings
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

PROMPT_SECRET = "KCHAT_PROMPT_MUST_NOT_REACH_SPEND_LOGS"
RESPONSE_SECRET = "KCHAT_RESPONSE_MUST_NOT_REACH_SPEND_LOGS"


def main() -> None:
    os.environ.pop("STORE_PROMPTS_IN_SPEND_LOGS", None)
    general_settings["store_prompts_in_spend_logs"] = False
    now = datetime.datetime.now(datetime.timezone.utc)
    response = litellm.ModelResponse(
        id="chatcmpl-redaction-contract",
        choices=[
            litellm.Choices(
                finish_reason="stop",
                index=0,
                message=litellm.Message(content=RESPONSE_SECRET, role="assistant"),
            )
        ],
        model="strict-local/test",
        usage=litellm.Usage(completion_tokens=2, prompt_tokens=2, total_tokens=4),
    )
    standard = {
        "messages": [{"role": "user", "content": PROMPT_SECRET}],
        "response": {"role": "assistant", "content": RESPONSE_SECRET},
        "metadata": {"user_api_key_end_user_id": "redaction-contract"},
    }
    payload = get_logging_payload(
        kwargs={
            "model": "strict-local/test",
            "messages": standard["messages"],
            "standard_logging_object": standard,
            "litellm_params": {
                "metadata": {"user_api_key": "test-key"},
                "proxy_server_request": {
                    "body": {
                        "model": "strict-local/test",
                        "messages": standard["messages"],
                    }
                },
            },
        },
        response_obj=response,
        start_time=now,
        end_time=now,
    )

    encoded = json.dumps(payload, default=str)
    assert PROMPT_SECRET not in encoded
    assert RESPONSE_SECRET not in encoded
    assert payload["messages"] == "{}"
    assert payload["response"] == "{}"
    assert payload["proxy_server_request"] == "{}"


if __name__ == "__main__":
    main()
