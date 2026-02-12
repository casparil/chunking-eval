from typing import Mapping, Dict, List, Any

import torch
from transformers import BatchEncoding, PreTrainedTokenizerFast


def move_to_cuda(sample: Any) -> Any:
    """
    Checks if the given input is a PyTorch tensor and moves it to the GPU. If the input has a nested structure, i.e. is
    a list, dictionary or similar, each element in this structure is checked.

    :param sample: The input to check.
    :return: The inputs, now possibly moved to GPU.
    """
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor: Any) -> Any:
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)(
                {k: _move_to_cuda(v) for k, v in maybe_tensor.items()}
            )
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor, pool_type: str
) -> torch.Tensor:
    """
    Applies pooling to the generated embeddings and returns the pooled results. Uses average or CLS pooling depending
    on the passed type.

    :param last_hidden_states: The model output including the generated embeddings.
    :param attention_mask: The attention mask to use.
    :param pool_type: The type of pooling to use.
    :return: The pooled embeddings.
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    else:
        raise ValueError(f"pool_type {pool_type} not supported")

    return emb


def transform_func(tokenizer: PreTrainedTokenizerFast, examples: Dict[str, List]) -> BatchEncoding:
    """
    Tokenizes the given text and returns the results.

    :param tokenizer: The tokenizer to use.
    :param examples: The dictionary containing the text to tokenize.
    :return: The tokenized text.
    """
    return tokenizer(
        examples["contents"],
        max_length=8192,
        padding=True,
        return_token_type_ids=False,
        truncation=True,
    )


def create_batch_dict(
    tokenizer: PreTrainedTokenizerFast, input_texts: List[str], max_length: int = 512
) -> BatchEncoding:
    """
    Tokenizes the given text and returns the padded tensors.

    :param tokenizer: The tokenizer to use.
    :param input_texts: The input text to tokenize.
    :param max_length: The maximum length of returned tokens.
    :return: The tokenized text.
    """
    return tokenizer(
        input_texts,
        max_length=max_length,
        padding=True,
        pad_to_multiple_of=8,
        return_token_type_ids=False,
        truncation=True,
        return_tensors="pt",
    )


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Applies pooling the generated embeddings and returns the pooled embeddings.

    :param last_hidden_states: The model output including the generated embeddings.
    :param attention_mask: The attention mask to use.
    :return: The pooled embeddings.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
