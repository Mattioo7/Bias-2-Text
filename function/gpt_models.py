GPT_MODELS = ("gpt-4o", "gpt-4o-mini", "gpt-5.6-luna")
REASONING_MODELS = ("gpt-5.6-luna",)


def completion_kwargs(model, max_tokens=300):
    """Output-length (and reasoning) arguments for chat.completions.create."""
    if model in REASONING_MODELS:
        # Reasoning models reject max_tokens. Reasoning tokens also count towards the limit,
        # so without effort "none" they can use up the whole budget and return an empty answer.
        return {"max_completion_tokens": max_tokens, "reasoning_effort": "none"}
    return {"max_tokens": max_tokens}
