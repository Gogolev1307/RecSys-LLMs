"""Prompt and frozen-cohort checks; no model responses are simulated here."""
import json
import unittest
from types import SimpleNamespace

import buy_sell_ranker
import ranker
from runtime import inference_dtype
from test_ranker import IDS, fixture


class BuySellPromptTests(unittest.TestCase):
    def test_only_sell_history_is_added_and_future_is_filtered(self):
        assets, context = fixture()
        events = [
            ('2021-05-31', IDS[2], 'Sell'), ('2021-05-31', IDS[1], 'Sell'),
            ('2021-05-29', IDS[1], 'Sell'), ('2021-05-30', IDS[2], 'Buy'),
            ('2021-06-01', IDS[2], 'Sell'), ('2021-06-02', IDS[2], 'Buy'),
            ('2021-05-31', IDS[0], 'Transfer'), ('2021-05-31', 'INELIGIBLE', 'Sell'),
        ]
        inputs = context, assets, {IDS[2]: 1}, events, 2
        prompt = buy_sell_ranker.build_prompt(*inputs)
        instructions, raw_payload = prompt.rsplit('\n', 1)
        self.assertIn('not a purchase or an automatic dislike', instructions)
        payload = json.loads(raw_payload)
        self.assertEqual(payload.pop('recent_sell_events_oldest_to_newest'), [
            {'date': '2021-05-31', 'ISIN': IDS[1]}, {'date': '2021-05-31', 'ISIN': IDS[2]},
        ])
        self.assertEqual(payload, ranker.prompt_payload(*inputs))

    def test_no_recorded_sells_is_empty(self):
        assets, context = fixture()
        prompt = buy_sell_ranker.build_prompt(context, assets, {}, ())
        payload = json.loads(prompt.rsplit('\n', 1)[1])
        self.assertEqual(payload['recent_sell_events_oldest_to_newest'], [])

    def test_smoke_selects_by_past_operations_only(self):
        events = {'first': [('2021-05-31', IDS[0], 'Buy')],
                  'second': [('2021-05-31', IDS[0], 'Buy'), ('2021-05-31', IDS[0], 'Sell')]}
        manifest = {'users': ['first', 'second']}
        bundle = (None, None, None, None, None, None, events, None, manifest)
        subset, index = buy_sell_ranker.smoke_bundle(bundle)
        self.assertEqual((subset[-1]['users'], index), (['second'], 1))
        self.assertEqual(manifest['users'], ['first', 'second'])


class DtypeTests(unittest.TestCase):
    @staticmethod
    def torch_device(*, cuda, bf16):
        return SimpleNamespace(
            float32='fp32', float16='fp16', bfloat16='bf16',
            cuda=SimpleNamespace(is_available=lambda: cuda,
                                 is_bf16_supported=lambda *, including_emulation: bf16),
        )

    def test_cpu_and_cuda_precision_selection(self):
        self.assertEqual(inference_dtype(self.torch_device(cuda=False, bf16=False),
                                         'cpu', 'bfloat16'), 'fp32')
        self.assertEqual(inference_dtype(self.torch_device(cuda=True, bf16=True),
                                         'cuda', 'bfloat16'), 'bf16')
        self.assertEqual(inference_dtype(self.torch_device(cuda=True, bf16=False),
                                         'cuda', 'float16'), 'fp16')

    def test_bfloat16_on_t4_and_missing_cuda_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, '--cuda-dtype float16'):
            inference_dtype(self.torch_device(cuda=True, bf16=False), 'cuda', 'bfloat16')
        with self.assertRaisesRegex(ValueError, 'CUDA requested but unavailable'):
            inference_dtype(self.torch_device(cuda=False, bf16=False), 'cuda', 'float16')
