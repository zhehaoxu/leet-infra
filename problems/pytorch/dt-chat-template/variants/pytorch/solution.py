"""Fixed ChatML rendering with UTF-8 byte annotations; standard library only."""


def solve(messages, add_generation_prompt=False, continue_final_message=False):
    if add_generation_prompt and continue_final_message:
        raise ValueError("generation and continuation are mutually exclusive")
    if continue_final_message and not messages:
        raise ValueError("cannot continue an empty conversation")

    role_numbers = {"system": 0, "user": 1, "assistant": 2}
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("each message must contain exactly role and content")
        role = message["role"]
        if not isinstance(role, str) or role not in role_numbers:
            raise ValueError("unknown role")
        if not isinstance(message["content"], str):
            raise ValueError("content must be a string")

    fragments = []
    input_bytes = []
    role_ids = []
    assistant_mask = []
    content_spans = []

    def append(fragment, role_id=-1):
        encoded = fragment.encode("utf-8")
        fragments.append(fragment)
        input_bytes.extend(encoded)
        role_ids.extend([role_id] * len(encoded))
        assistant_mask.extend([role_id == 2] * len(encoded))

    for index, message in enumerate(messages):
        role = message["role"]
        append("<|im_start|>" + role + "\n")
        start = len(input_bytes)
        append(message["content"], role_numbers[role])
        content_spans.append((start, len(input_bytes)))
        if not (continue_final_message and index == len(messages) - 1):
            append("<|im_end|>\n")

    if add_generation_prompt:
        append("<|im_start|>assistant\n")

    return {
        "text": "".join(fragments),
        "input_bytes": input_bytes,
        "role_ids": role_ids,
        "assistant_mask": assistant_mask,
        "content_spans": content_spans,
    }
