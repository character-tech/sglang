"""Serving test for MIXED-batch split dispatch on the Triton backend.

Launches a hybrid-SWA Gemma4 model with --enable-mixed-chunk and a small
chunked-prefill-size so that continuous concurrent traffic makes most steps
MIXED (running decodes folded into every prefill batch). With split dispatch
(SGLANG_TRITON_MIXED_SPLIT_DISPATCH, default on) the decode suffix runs the
grouped decode kernels against decode-mode metadata built inside a MIXED
step — this test exercises that metadata path (windowed SWA indices over the
suffix rows, split-KV, whole-batch KV save ordering) under load, where an
indexing bug would surface as an illegal memory access, a hang, or garbage
output ending generation early.

CPU-side split logic is covered by
test/registered/unit/layers/attention/test_triton_mixed_split_dispatch.py.
"""

import concurrent.futures
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=110, stage="base-b", runner_config="2-gpu-large")
register_amd_ci(est_time=160, suite="stage-b-test-2-gpu-large-amd")


# Long enough (with the repeats below) to span several 256-token prefill
# chunks, so chunked prefill and mixing both engage.
PROMPT = (
    "Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast "
    "every morning and bakes muffins for her friends every day with four. She "
    "sells the remainder at the farmers' market daily for $2 per fresh duck "
    "egg. How much in dollars does she make every day at the farmers' market? "
) * 8 + "\nAnswer:"
NUM_REQUESTS = 120
CONCURRENCY = 64
MAX_TOKENS = 128


class TestGemma4MixedChunkSplit(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "google/gemma-4-26B-A4B-it"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "2",
                "--attention-backend",
                "triton",
                "--enable-mixed-chunk",
                "--chunked-prefill-size",
                "256",
                "--dtype",
                "bfloat16",
                "--mem-fraction-static",
                "0.55",
                "--max-running-requests",
                "16",
                "--context-length",
                "4096",
                "--max-total-tokens",
                "32768",
                "--skip-server-warmup",
                "--random-seed",
                "0",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _fire_one(self):
        try:
            r = requests.post(
                self.base_url + "/v1/completions",
                json={
                    "model": self.model,
                    "prompt": PROMPT,
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "top_k": 1,
                },
                timeout=300,
            )
            r.raise_for_status()
            body = r.json()
            n_tokens = body["usage"]["completion_tokens"]
            if n_tokens < 1:
                return False, f"empty completion: {body!r}"
            return True, ""
        except Exception as e:
            return False, repr(e)

    def test_mixed_chunk_split_under_concurrent_load(self):
        try:
            requests.get(self.base_url + "/flush_cache", timeout=30)
        except Exception:
            pass

        n_ok = n_fail = 0
        first_fail = ""
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futs = [ex.submit(self._fire_one) for _ in range(NUM_REQUESTS)]
            for f in concurrent.futures.as_completed(futs):
                ok, msg = f.result()
                if ok:
                    n_ok += 1
                else:
                    if n_fail == 0:
                        first_fail = msg
                    n_fail += 1

        print(f"n_ok={n_ok} n_fail={n_fail} first_fail={first_fail!r}")
        self.assertEqual(
            n_fail,
            0,
            f"{n_fail}/{NUM_REQUESTS} requests failed; first error: {first_fail}",
        )


if __name__ == "__main__":
    unittest.main()
