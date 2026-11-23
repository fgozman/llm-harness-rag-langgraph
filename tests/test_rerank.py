from rag.retrieval.rerank import _QWEN_PREFIX, _QWEN_SUFFIX, format_qwen_pair, is_qwen3_reranker  # pyright: ignore[reportPrivateUsage]


def test_qwen3_prompt_pieces_match_model_card_verbatim() -> None:
    assert _QWEN_PREFIX == (
        "<|im_start|>system\nJudge whether the Document meets the requirements "
        "based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    assert _QWEN_SUFFIX == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert format_qwen_pair("I", "Q", "D") == "<Instruct>: I\n<Query>: Q\n<Document>: D"


def test_reranker_family_detection() -> None:
    assert is_qwen3_reranker("Qwen/Qwen3-Reranker-0.6B")
    assert not is_qwen3_reranker("BAAI/bge-reranker-v2-m3")
