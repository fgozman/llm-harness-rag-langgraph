import threading

import pytest

from rag.config import ModelPricing
from rag.llm.usage import UsageMeter

PRICING = {
    "cheap": ModelPricing(input=1.0, cache_write_5m=1.25, cache_write_1h=2.0, cache_read=0.1, output=5.0),
    "big": ModelPricing(input=10.0, cache_write_5m=12.5, cache_write_1h=20.0, cache_read=1.0, output=50.0),
}


def test_records_per_model_and_computes_cost() -> None:
    meter = UsageMeter()
    meter.record("cheap", input_tokens=1000, output_tokens=100)
    meter.record("cheap", input_tokens=500, output_tokens=50, cache_read_tokens=2000)
    meter.record("big", input_tokens=100, output_tokens=10, cache_write_tokens=1000)
    snap = meter.snapshot()
    assert snap["cheap"].input_tokens == 1500 and snap["cheap"].cache_read_tokens == 2000
    assert snap["cheap"].calls == 2 and snap["big"].calls == 1
    # cheap: 1500*1 + 150*5 + 2000*0.1 = 2450; big: 100*10 + 10*50 + 1000*12.5 = 14000  (per 1M tokens)
    assert meter.cost_usd(PRICING) == pytest.approx((2450 + 14000) / 1e6)


def test_unknown_model_costs_nothing_but_is_counted() -> None:
    meter = UsageMeter()
    meter.record("local-llama", input_tokens=10, output_tokens=5)
    assert meter.cost_usd(PRICING) == 0.0
    assert meter.total_tokens() == 15


def test_reset_and_thread_safety() -> None:
    meter = UsageMeter()

    def work() -> None:
        for _ in range(1000):
            meter.record("cheap", input_tokens=1, output_tokens=1)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert meter.snapshot()["cheap"].calls == 8000
    meter.reset()
    assert meter.snapshot() == {}
