import os

import pytest
import torch

from rag import devices


@pytest.fixture
def hw(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    return monkeypatch


def test_auto_prefers_cuda_then_mps_then_cpu(hw: pytest.MonkeyPatch) -> None:
    assert devices.resolve_device("auto") == "cpu"
    hw.setattr(torch.backends.mps, "is_available", lambda: True)
    assert devices.resolve_device("auto") == "mps"
    hw.setattr(torch.cuda, "is_available", lambda: True)
    assert devices.resolve_device("auto") == "cuda"


def test_explicit_unavailable_device_raises(hw: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="cuda"):
        devices.resolve_device("cuda")
    assert devices.resolve_device("cpu") == "cpu"


@pytest.mark.parametrize(
    ("device", "requested", "expected"),
    [
        ("mps", "bfloat16", torch.float16),  # M1 has no native bfloat16
        ("mps", "float16", torch.float16),
        ("cpu", "float16", torch.float32),  # half precision on CPU is slow/unsupported
        ("cuda", "bfloat16", torch.bfloat16),
        ("cpu", "float32", torch.float32),
    ],
)
def test_dtype_is_adjusted_to_device(device: str, requested: str, expected: torch.dtype) -> None:
    assert devices.resolve_dtype(device, requested) == expected  # type: ignore[arg-type]


def test_attention_impl_is_never_flash_attention() -> None:
    assert devices.ATTN_IMPLEMENTATION == "sdpa"


def test_mps_fallback_env_is_set_on_import() -> None:
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
