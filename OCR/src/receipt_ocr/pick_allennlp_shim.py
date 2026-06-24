"""Minimal allennlp stubs so MC_OCR PICK CRF loads on modern PyTorch (no allennlp 1.0)."""
from __future__ import annotations

import sys
import types
from typing import List, Tuple

import torch


class ConfigurationError(Exception):
    pass


def logsumexp(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    return torch.logsumexp(tensor, dim=dim, keepdim=keepdim)


def viterbi_decode(
    tag_sequence: torch.Tensor,
    transition_matrix: torch.Tensor,
) -> Tuple[List[int], torch.Tensor]:
    """Viterbi on a single sequence (allennlp-compatible subset)."""
    sequence_size, num_tags = tag_sequence.shape
    viterbi = tag_sequence[0].reshape(num_tags, 1)
    backpointers: list[torch.Tensor] = []

    for i in range(1, sequence_size):
        broadcast = viterbi.expand(num_tags, num_tags)
        summed = broadcast + transition_matrix
        max_scores, bp = summed.max(0)
        viterbi = max_scores + tag_sequence[i].reshape(num_tags, 1)
        backpointers.append(bp)

    best_path = [int(viterbi.argmax())]
    for bp in reversed(backpointers):
        best_path.append(int(bp[best_path[-1]]))
    best_path.reverse()
    best_score = viterbi.max()
    return best_path, best_score


def install_allennlp_shim() -> None:
    if "allennlp" in sys.modules:
        return

    checks = types.ModuleType("allennlp.common.checks")
    checks.ConfigurationError = ConfigurationError

    nn_util = types.ModuleType("allennlp.nn.util")
    nn_util.logsumexp = logsumexp
    nn_util.viterbi_decode = viterbi_decode
    nn_util.get_lengths_from_binary_sequence_mask = lambda mask: mask.sum(-1)

    nn_mod = types.ModuleType("allennlp.nn")
    nn_mod.util = nn_util

    span_utils = types.ModuleType("allennlp.data.dataset_readers.dataset_utils.span_utils")

    class InvalidTagSequence(Exception):
        pass

    span_utils.InvalidTagSequence = InvalidTagSequence

    metrics = types.ModuleType("allennlp.training.metrics.metric")

    class Metric:
        pass

    metrics.Metric = Metric

    common = types.ModuleType("allennlp.common")
    common.checks = checks

    data = types.ModuleType("allennlp.data")
    dataset_readers = types.ModuleType("allennlp.data.dataset_readers")
    dataset_readers.dataset_utils = types.ModuleType("allennlp.data.dataset_readers.dataset_utils")
    dataset_readers.dataset_utils.span_utils = span_utils
    data.dataset_readers = dataset_readers

    training = types.ModuleType("allennlp.training")
    training.metrics = types.ModuleType("allennlp.training.metrics")
    training.metrics.metric = metrics

    root = types.ModuleType("allennlp")
    root.common = common
    root.nn = nn_mod
    root.data = data
    root.training = training

    sys.modules["allennlp"] = root
    sys.modules["allennlp.common"] = common
    sys.modules["allennlp.common.checks"] = checks
    sys.modules["allennlp.nn"] = nn_mod
    sys.modules["allennlp.nn.util"] = nn_util
    sys.modules["allennlp.data"] = data
    sys.modules["allennlp.data.dataset_readers"] = dataset_readers
    sys.modules["allennlp.data.dataset_readers.dataset_utils"] = dataset_readers.dataset_utils
    sys.modules["allennlp.data.dataset_readers.dataset_utils.span_utils"] = span_utils
    sys.modules["allennlp.training"] = training
    sys.modules["allennlp.training.metrics"] = training.metrics
    sys.modules["allennlp.training.metrics.metric"] = metrics
