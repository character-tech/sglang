"""Unit tests for the Triton backend's MIXED-batch split dispatch.

Covers the CPU-side logic: the 1-token-suffix partition, the prefill metadata
view slicing, the fallback conditions, and forward_mixed's orchestration
(metadata swap, save_kv_cache=False on the decode sub-call, output stitching).
The kernels themselves are exercised by the GPU test
test/registered/attention/test_gemma4_mixed_chunk_split.py.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

import torch

from sglang.srt.layers.attention.triton_backend import (
    ForwardMetadata,
    MixedSplitMetadata,
    TritonAttnBackend,
    mixed_decode_suffix_start,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _forward_metadata(**overrides) -> ForwardMetadata:
    fields = dict(
        attn_logits=None,
        attn_lse=None,
        max_extend_len=None,
        num_kv_splits=None,
        kv_indptr=None,
        kv_indices=None,
        qo_indptr=None,
        custom_mask=None,
        mask_indptr=None,
        window_kv_indptr=None,
        window_kv_indices=None,
        window_num_kv_splits=None,
        window_kv_offsets=None,
    )
    fields.update(overrides)
    return ForwardMetadata(**fields)


class TestMixedDecodeSuffixStart(unittest.TestCase):
    def test_no_one_token_suffix(self):
        self.assertEqual(mixed_decode_suffix_start([512, 300, 7]), 3)

    def test_plain_mixed_batch(self):
        self.assertEqual(mixed_decode_suffix_start([512, 300, 1, 1, 1]), 2)

    def test_one_token_prefill_row_joins_suffix(self):
        # A chunked request's final 1-token chunk abutting the decode suffix
        # is decode-equivalent and classified into it.
        self.assertEqual(mixed_decode_suffix_start([512, 1, 1, 1]), 1)

    def test_interior_one_token_row_stays_prefill(self):
        self.assertEqual(mixed_decode_suffix_start([512, 1, 300, 1]), 3)

    def test_all_ones_keeps_first_row_for_extend(self):
        # The extend sub-call owns the whole-batch KV save; it keeps >= 1 row.
        self.assertEqual(mixed_decode_suffix_start([1, 1, 1]), 1)

    def test_single_row(self):
        self.assertEqual(mixed_decode_suffix_start([1]), 1)
        self.assertEqual(mixed_decode_suffix_start([64]), 1)


class TestBuildMixedSplitMetadata(unittest.TestCase):
    def _backend_with_metadata(self, metadata: ForwardMetadata) -> TritonAttnBackend:
        backend = TritonAttnBackend.__new__(TritonAttnBackend)
        backend.forward_metadata = metadata
        return backend

    def _extend_metadata(self, bs: int, with_window: bool) -> ForwardMetadata:
        return _forward_metadata(
            max_extend_len=999,
            qo_indptr=torch.arange(bs + 1, dtype=torch.int64),
            kv_indptr=torch.arange(bs + 1, dtype=torch.int32),
            kv_indices=torch.zeros(64, dtype=torch.int64),
            window_kv_indptr=(
                torch.arange(bs + 1, dtype=torch.int32) if with_window else None
            ),
            window_kv_offsets=torch.zeros(bs, dtype=torch.int32),
        )

    def test_missing_cpu_lens_returns_none(self):
        backend = self._backend_with_metadata(self._extend_metadata(3, False))
        fb = SimpleNamespace(extend_seq_lens_cpu=None, extend_prefix_lens_cpu=[0] * 3)
        self.assertIsNone(backend._build_mixed_split_metadata(fb))
        fb = SimpleNamespace(extend_seq_lens_cpu=[4, 1, 1], extend_prefix_lens_cpu=None)
        self.assertIsNone(backend._build_mixed_split_metadata(fb))

    def test_no_decode_suffix_returns_none(self):
        backend = self._backend_with_metadata(self._extend_metadata(2, False))
        fb = SimpleNamespace(
            extend_seq_lens_cpu=[512, 300], extend_prefix_lens_cpu=[0, 100]
        )
        self.assertIsNone(backend._build_mixed_split_metadata(fb))

    def test_prefill_view_slicing(self):
        bs = 5
        metadata = self._extend_metadata(bs, with_window=True)
        backend = self._backend_with_metadata(metadata)
        fb = SimpleNamespace(
            extend_seq_lens_cpu=[512, 300, 1, 1, 1],
            extend_prefix_lens_cpu=[0, 100, 900, 1500, 2000],
        )
        decode_metadata = _forward_metadata()
        with patch.object(
            TritonAttnBackend,
            "_build_mixed_decode_metadata",
            return_value=decode_metadata,
        ) as build_decode:
            mixed = backend._build_mixed_split_metadata(fb)

        self.assertIsInstance(mixed, MixedSplitMetadata)
        self.assertEqual(mixed.prefill_token_end, 812)
        # Row-indexed indptrs sliced to the 2 prefill rows; flat tensors shared.
        self.assertEqual(mixed.prefill.qo_indptr.shape[0], 3)
        self.assertEqual(mixed.prefill.kv_indptr.shape[0], 3)
        self.assertEqual(mixed.prefill.window_kv_indptr.shape[0], 3)
        self.assertIs(mixed.prefill.kv_indices, metadata.kv_indices)
        self.assertIs(mixed.prefill.window_kv_offsets, metadata.window_kv_offsets)
        self.assertEqual(mixed.prefill.max_extend_len, 512)
        self.assertIsNone(mixed.prefill.mixed_split)
        self.assertIs(mixed.decode, decode_metadata)
        # The step metadata itself is untouched (views, not mutation).
        self.assertEqual(metadata.qo_indptr.shape[0], bs + 1)
        self.assertEqual(metadata.max_extend_len, 999)
        kwargs = build_decode.call_args.kwargs
        self.assertEqual(kwargs["n_prefill"], 2)
        self.assertEqual(kwargs["n_decode"], 3)


class TestForwardMixed(unittest.TestCase):
    def _backend(self) -> TritonAttnBackend:
        backend = TritonAttnBackend.__new__(TritonAttnBackend)
        backend._mixed_split_logged = True
        return backend

    def test_fallback_folds_into_extend(self):
        backend = self._backend()
        backend.forward_metadata = _forward_metadata(mixed_split=None)
        q = torch.zeros(4, 8)
        with patch.object(
            TritonAttnBackend, "forward_extend", return_value=q
        ) as extend:
            out = backend.forward_mixed(q, q, q, layer=None, forward_batch=None)
        self.assertIs(out, q)
        extend.assert_called_once()

    def test_split_path_stitches_outputs(self):
        backend = self._backend()
        prefill_view = _forward_metadata(qo_indptr=torch.zeros(3, dtype=torch.int64))
        decode_view = _forward_metadata(kv_indptr=torch.zeros(3, dtype=torch.int32))
        full = _forward_metadata(
            mixed_split=MixedSplitMetadata(
                prefill=prefill_view, decode=decode_view, prefill_token_end=6
            )
        )
        backend.forward_metadata = full
        n_tokens, n_decode, dim = 8, 2, 4
        q = torch.zeros(n_tokens, dim)
        seen = {}

        def fake_extend(self, *args, **kwargs):
            seen["extend_metadata"] = self.forward_metadata
            return torch.full((n_tokens, dim), 1.0)

        def fake_decode(self, q_dec, k_dec, v_dec, layer, fb, **kwargs):
            seen["decode_metadata"] = self.forward_metadata
            seen["decode_q_rows"] = q_dec.shape[0]
            seen["decode_save_kv_cache"] = kwargs["save_kv_cache"]
            return torch.full((n_decode, dim), 2.0)

        with patch.object(TritonAttnBackend, "forward_extend", fake_extend):
            with patch.object(TritonAttnBackend, "forward_decode", fake_decode):
                out = backend.forward_mixed(q, q, q, layer=None, forward_batch=None)

        self.assertIs(seen["extend_metadata"], prefill_view)
        self.assertIs(seen["decode_metadata"], decode_view)
        self.assertEqual(seen["decode_q_rows"], n_decode)
        self.assertFalse(seen["decode_save_kv_cache"])
        self.assertTrue(torch.equal(out[:6], torch.full((6, dim), 1.0)))
        self.assertTrue(torch.equal(out[6:], torch.full((2, dim), 2.0)))
        # Metadata restored after both sub-calls.
        self.assertIs(backend.forward_metadata, full)


if __name__ == "__main__":
    unittest.main()
