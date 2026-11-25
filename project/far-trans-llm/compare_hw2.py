"""Report the measured HW2 results without modifying either run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from buy_sell_ranker import BASELINE, MODEL, REVISION


BASE_METHODS = {
    'popularity': 'Popularity',
    'cbf': 'CBF',
    'latest_buy': 'Latest Buy',
    'latest_buy_or_sell': 'Latest Buy or Sell',
}
TREATMENT = 'llmrank_buy_sell'


def load(directory, name):
    return json.loads((directory / name).read_text(encoding='utf-8'))


def json_lines(directory, name):
    return [json.loads(line) for line in (directory / name).read_text(encoding='utf-8').splitlines()]


def csv_rows(directory, name):
    with (directory / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prediction_rows(directory, methods=BASE_METHODS):
    return {(r['user_id'], r['method'], int(r['rank'])): r['ISIN']
            for r in csv_rows(directory, 'predictions.csv') if r['method'] in methods}


def results(baseline_dir, llm_dir):
    protocol = load(baseline_dir, 'protocol.json')
    baseline = load(baseline_dir, 'metrics.json')
    require(protocol['protocol_hash'] == baseline['protocol_hash'], 'Baseline protocol hash mismatch')
    if llm_dir is None:
        return baseline['methods'], None

    experiment = load(llm_dir, 'experiment.json')
    require(experiment['status'] == 'evaluated' and experiment['mode'] == 'full'
            and experiment['history_mode'] == 'buy-sell', 'Expected a completed BUY/SELL full run')
    require((experiment['model'], experiment['requested_revision']) == (MODEL, REVISION),
            'Unexpected model or revision')
    manifest = load(llm_dir, 'run_manifest.json')
    context_hash = hashlib.sha256((baseline_dir / 'contexts.jsonl').read_bytes()).hexdigest()
    require(manifest['baseline_protocol_hash'] == protocol['protocol_hash']
            and manifest['frozen_contexts_sha256'] == context_hash
            and manifest['data_sha256'] == protocol['source_sha256']
            and all(manifest[key] == protocol[key] for key in
                    ('users', 'cutoff', 'end_exclusive', 'candidate_count')),
            'LLM protocol differs from the saved baselines')
    require(prediction_rows(baseline_dir) == prediction_rows(llm_dir), 'Classical rankings differ')
    metrics = load(llm_dir, 'metrics.json')
    for method in BASE_METHODS:
        old, new = baseline['methods'][method], metrics[method]
        require(old.keys() == new.keys() and all(
            math.isclose(old[key], new[key], rel_tol=1e-12, abs_tol=1e-15) for key in old
        ), f'Classical metrics differ for {method}')

    labels = {r['user_id']: int(r['relevant_count']) for r in csv_rows(baseline_dir, 'per_user.csv')
              if r['method'] == 'latest_buy_or_sell'}
    per_user = [r for r in load(llm_dir, 'per_user.json') if r['method'] == TREATMENT]
    require(len(per_user) == len(labels)
            and {r['user_id']: r['relevant_count'] for r in per_user} == labels,
            'Per-user evaluation truth differs')
    responses = json_lines(llm_dir, 'final_responses.jsonl')
    require([r['user_id'] for r in responses] == protocol['users'], 'LLM response coverage differs')
    candidates = {r['user_id']: [asset['id'] for asset in r['candidate_assets']]
                  for r in json_lines(baseline_dir, 'contexts.jsonl')}
    expected = {}
    for r in responses:
        uid, ids = r['user_id'], r['ranked_ids']
        allowed = candidates[uid]
        require(len(ids) == len(set(ids)) == 10 and set(ids) <= set(allowed),
                'Ranked IDs are not ten unique fixed candidates')
        require((r['fallback'] and r['error'] and ids == allowed[:10])
                or (not r['fallback'] and r['error'] is None), 'Unexpected fallback behavior')
        expected.update({(uid, TREATMENT, rank): isin for rank, isin in enumerate(ids, 1)})
    require(expected == prediction_rows(llm_dir, (TREATMENT,)), 'Final LLM rankings differ from predictions')
    fallback = sum(r['fallback'] for r in responses)
    failures = len(manifest['failures'])
    require(fallback == failures == manifest['runtime']['raw_invalid_count'],
            'Fallback and failure logs do not agree')
    valid_users = metrics[TREATMENT]['users'] - fallback
    return baseline['methods'], (metrics[TREATMENT], experiment['wall_seconds'], failures, valid_users)


def smoke_timing(directory):
    experiment = load(directory, 'experiment.json')
    print(f"\nSmoke status: {experiment['status']}")
    generation_path = directory / 'generations.jsonl'
    if not generation_path.exists():
        return
    rows = [r for r in json_lines(directory, generation_path.name) if not r['cache_hit']]
    require(len(rows) == 1, 'Expected exactly one fresh smoke generation')
    response = rows[0]['response']
    seconds = response['seconds']
    print(f"One-user inference: {seconds:.2f} s "
          f"({response['input_tokens']} input tokens, {response['output_tokens']} output tokens).")
    if experiment['status'] == 'valid_real_inference':
        print(f"100-user generation estimate: {seconds * 100 / 60:.1f} min "
              "(one sample × 100; excludes loading and grading; token lengths vary).")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, default=BASELINE)
    parser.add_argument('--llm-dir', type=Path, help='Completed 100-user BUY/SELL output')
    parser.add_argument('--smoke-dir', type=Path)
    parser.add_argument('--baseline-wall-seconds', type=float, default=3.80,
                        help='Measured combined baseline runtime (default: original local run)')
    args = parser.parse_args(argv)
    try:
        classical, llm = results(args.baseline_dir, args.llm_dir)
        print('| Method | Users | Users with future purchases | nDCG@10 all | nDCG@10 nonempty | Runtime | Failed cases |')
        print('|---|---:|---:|---:|---:|---|---:|')
        def row(name, metrics, runtime, failures):
            print(f"| {name} | {metrics['users']} | {metrics['nonempty_truth_users']} | "
                  f"{metrics['ndcg_at_10']:.4f} | {metrics['ndcg_at_10_nonempty']:.4f} | "
                  f"{runtime} | {failures} |")
        for method, name in BASE_METHODS.items():
            row(name, classical[method], 'See shared runtime', 0)
        print_row = '| BUY/SELL Ranker | — | — | — | — | Not evaluated | — |'
        if llm is None:
            print(print_row)
        else:
            metric, seconds, failures, valid_users = llm
            row('BUY/SELL Ranker', metric, f'{seconds:.2f} s (full run)', failures)
        print(f'\nCombined four-baseline wall time: {args.baseline_wall_seconds:.2f} s.')
        if llm is not None:
            latest = classical['latest_buy_or_sell']
            print(f"Valid without fallback: {valid_users}/{metric['users']}. "
                  "Ranker averages include fallback cases.")
            print(f"Delta vs Latest Buy or Sell: "
                  f"{metric['ndcg_at_10'] - latest['ndcg_at_10']:+.4f} (all), "
                  f"{metric['ndcg_at_10_nonempty'] - latest['ndcg_at_10_nonempty']:+.4f} (nonempty).")
        if args.smoke_dir:
            smoke_timing(args.smoke_dir)
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(1, f'Cannot report results: {exc}\n')


if __name__ == '__main__':
    main()
