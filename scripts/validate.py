# pyre-unsafe
"""Validation harness for Sturnus."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
import argparse
from typing import Dict, List, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experts import ExpertPool
from gating import GateRouter


SAMPLE_PROMPTS: List[str] = [
    "hello world",
    "explain eigenvectors in hilbert space",
    "write a python function to reverse a list",
    "summarize the major steps in a data pipeline",
    "what is photosynthesis",
    "design a cache eviction strategy",
    "prove that the sum of two even numbers is even",
    "generate a JSON schema for a user profile",
    "how to fix a memory leak in python",
    "describe transformer attention",
]


async def _run_end_to_end(router: GateRouter, prompt: str, context: str) -> Dict[str, Any]:
    decision = router.route(prompt, context)
    if decision.timeline == "A":
        result = await router.run_timeline_a(prompt)
    else:
        result = await router.run_timeline_b(
            prompt, context, decision.expert_indices, mode=2, x_concurrency=decision.x
        )
    return {"decision": decision, "result": result}


def run(samples: int = 50) -> None:
    print("[validate] running validation")
    pool = ExpertPool(max_cache=64)
    router = GateRouter(pool)

    ks: List[int] = []
    fast_path = 0
    timeline_b = 0

    print("[validate] Phase 1: routing distribution")
    for i in range(samples):
        prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
        decision = router.route(prompt, context="")
        ks.append(decision.k)
        if decision.timeline == "A":
            fast_path += 1
        else:
            timeline_b += 1
        pool.update_utilization(decision.expert_indices)

    k_arr = np.array(ks)
    print(f"[validate] fast-path rate: {fast_path / samples:.2f}")
    print(f"[validate] timeline-B rate: {timeline_b / samples:.2f}")
    print(f"[validate] K mean/min/max: {k_arr.mean():.2f}/{k_arr.min()}/{k_arr.max()}")
    utilization = pool.utilization_rates()
    print(f"[validate] utilization min/max: {min(utilization):.4f}/{max(utilization):.4f}")
    active_experts = sum(1 for u in utilization if u > 0)
    print(f"[validate] active experts: {active_experts}/{len(utilization)}")

    print("[validate] Phase 2: end-to-end execution")
    test_prompts = SAMPLE_PROMPTS[:3]

    async def _run_e2e():
        for prompt in test_prompts:
            output = await _run_end_to_end(router, prompt, context="")
            d = output["decision"]
            mode = output["result"].get("mode", "?")
            has_synth = "synth_vector" in output["result"]
            print(
                f"[validate] '{prompt[:30]}...' -> K={d.k} X={d.x} Y={d.y} "
                f"timeline={d.timeline} mode={mode} synth={has_synth}"
            )

    asyncio.run(_run_e2e())

    print("[validate] Phase 3: anti-collapse check")
    underused = pool.least_used()
    print(f"[validate] underused experts: {len(underused)}")
    print(f"[validate] EMA fast-path rate: {router.ema.fast_path_rate:.3f}")
    print(f"[validate] EMA entropy: {router.ema.entropy:.3f}")
    print(f"[validate] fast_path_threshold: {router.fast_path_threshold:.3f}")

    print("[validate] DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    run(samples=args.samples)
