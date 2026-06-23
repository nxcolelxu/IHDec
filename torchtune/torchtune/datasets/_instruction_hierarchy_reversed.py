# Reversed-order evaluation dataset for both_conflict_default_role_eval.json
# Original order: system → user1 → assistant → user2 → data
# Reversed order: data  → user2 → assistant → user1 → system

from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
from torch.utils.data import Dataset
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX, Message
from torchtune.datasets._instruction_hierarchy import load_IH_dataset, InstructHierarchyDataset
from torchtune.modules.tokenizers import ModelTokenizer


def message_converter_eval_5msg(sample: Mapping[str, Any], train_on_input: bool) -> List[Message]:
    """5-message eval converter (original order): system, user1, assistant, user2, data → empty assistant."""
    roles_order = [
        ("system",    sample[0]['content']),
        ("user",      sample[1]['content']),
        ("assistant", sample[2]['content']),
        ("user",      sample[3]['content']),
        ("data",      sample[4]['content']),
    ]
    messages = []
    for role, content in roles_order:
        if not content.strip() and role == "data":
            continue
        messages.append(Message(role=role, content=content, masked=False))
    messages.append(Message(role="assistant", content="", masked=False))
    return messages


def message_converter_eval_5msg_reversed(sample: Mapping[str, Any], train_on_input: bool) -> List[Message]:
    """5-message reversed eval converter: data, user2, assistant, user1, system → empty assistant."""
    # Original order: [system(0), user1(1), assistant(2), user2(3), data(4)]
    # Reversed order: [data(4), user2(3), assistant(2), user1(1), system(0)]
    roles_order = [
        ("data",      sample[4]['content']),
        ("user",      sample[3]['content']),
        ("assistant", sample[2]['content']),
        ("user",      sample[1]['content']),
        ("system",    sample[0]['content']),
    ]
    messages = []
    for role, content in roles_order:
        if not content.strip() and role == "data":
            continue
        messages.append(Message(role=role, content=content, masked=False))
    messages.append(Message(role="assistant", content="", masked=False))
    return messages


def InstructHierarchyVal5msg(
    *,
    tokenizer: ModelTokenizer,
    source: str,
    train_on_input: bool = False,
    max_seq_len: Optional[int] = None,
    packed: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> InstructHierarchyDataset:
    """Original 5-message order dataset for eval."""
    ds = InstructHierarchyDataset(
        tokenizer=tokenizer,
        source=source,
        convert_to_messages=message_converter_eval_5msg,
        train_on_input=train_on_input,
        max_seq_len=max_seq_len,
        **load_dataset_kwargs,
    )
    return ds


def InstructHierarchyValReversed(
    *,
    tokenizer: ModelTokenizer,
    source: str,
    train_on_input: bool = False,
    max_seq_len: Optional[int] = None,
    packed: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> InstructHierarchyDataset:
    """Reversed 5-message order dataset for eval (data→user2→assistant→user1→system)."""
    ds = InstructHierarchyDataset(
        tokenizer=tokenizer,
        source=source,
        convert_to_messages=message_converter_eval_5msg_reversed,
        train_on_input=train_on_input,
        max_seq_len=max_seq_len,
        **load_dataset_kwargs,
    )
    return ds
