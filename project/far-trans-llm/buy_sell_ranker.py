"""HW2 ranker using a separate, strictly pre-cutoff SELL history."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from argparse import Namespace
from pathlib import Path

import data_adapter
import ranker


MODEL = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
BASELINE = Path(__file__).resolve().parent.parent / 'far-trans' / 'outputs' / 'hw2_baseline'


def build_prompt(context, assets, history, events, history_events=15, shuffle_seed=None):
    """Keep the provided BUY prompt and append only the separate SELL evidence."""
    instructions, payload = ranker.build_prompt(
        context, assets, history, events, history_events, shuffle_seed).split('\n', 1)
    payload = json.loads(payload)
    cutoff = context['cutoff']
    sells = sorted((day[:10], isin) for day, isin, kind in events
                   if kind == 'Sell' and day[:10] < cutoff and isin in assets)
    payload['recent_sell_events_oldest_to_newest'] = [
        {'date': day, 'ISIN': isin} for day, isin in sells[-history_events:]]
    return (instructions + '\nBUY activity is in buy_history_counts and recent_buy_events_oldest_to_newest. '
            'SELL activity is separate in recent_sell_events_oldest_to_newest; each row is a SELL, '
            'not a purchase or an automatic dislike. SELL observations are sorted by date then ISIN '
            '(no intraday order is known). An empty list means no recorded SELL in the supplied history.\n'
            + json.dumps(payload, ensure_ascii=False, sort_keys=True))


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_candidates(bundle, baseline_dir):
    """Verify the homework split and use the original CBF candidates in saved order."""
    _, assets, _, _, contexts, baseline, _, _, manifest = bundle
    protocol = json.loads((baseline_dir / 'protocol.json').read_text())
    digest = protocol.pop('protocol_hash')
    if digest != hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest():
        raise ValueError('Saved baseline protocol hash does not match its contents')
    if (protocol['cutoff'] != '2021-06-01' or protocol['end_exclusive'] != '2021-07-01'
            or protocol['cohort_selection'] != 'warm' or len(protocol['users']) != 100
            or protocol['exclude_seen'] or protocol['candidate_count'] != 50 or protocol['k'] != 10
            or protocol['eligible_assets'] != sorted(assets)
            or protocol['source_sha256'] != manifest['data_sha256']
            or any(protocol[key] != manifest[key] for key in
                   ('cutoff', 'end_exclusive', 'users', 'candidate_count'))):
        raise ValueError('Run does not match the frozen Homework 2 protocol')
    context_path = baseline_dir / 'contexts.jsonl'
    saved = [json.loads(line) for line in context_path.read_text().splitlines()]
    if [c['user_id'] for c in saved] != manifest['users']:
        raise ValueError('Saved candidate users or order do not match the protocol')
    eligible = set(assets)
    for context in saved:
        uid = context['user_id']
        ids = ranker.candidate_ids(context)
        if (context['cutoff'] != manifest['cutoff'] or len(ids) != 50
                or not set(ids) <= eligible):
            raise ValueError('Saved candidate list does not match the protocol')
        # CBF can have floating-point ties; freeze presentation and its top ten.
        contexts[uid]['candidate_assets'] = [{'id': isin} for isin in ids]
        baseline[uid]['cbf'] = ids[:10]
    manifest.update(baseline_protocol_hash=digest, frozen_contexts_sha256=file_digest(context_path))


def smoke_bundle(bundle):
    """Select the first frozen user with a past BUY and SELL, without using labels."""
    events, manifest = bundle[6], bundle[-1]
    index, uid = next((i, uid) for i, uid in enumerate(manifest['users'])
                      if {'Buy', 'Sell'} <= {kind for _, _, kind in events.get(uid, ())})
    return (*bundle[:-1], dict(manifest, users=[uid])), index


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def environment():
    dependencies = {}
    for name in ('torch', 'transformers', 'huggingface-hub', 'tokenizers', 'safetensors'):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    details = {'platform': platform.platform(), 'machine': platform.machine(),
               'python': platform.python_version(), 'dependencies': dependencies}
    if platform.system() == 'Darwin':
        for key, field in (('machdep.cpu.brand_string', 'cpu'), ('hw.memsize', 'memory_bytes')):
            value = subprocess.run(['/usr/sbin/sysctl', '-n', key], capture_output=True, text=True)
            if value.returncode == 0:
                details[field] = int(value.stdout) if field == 'memory_bytes' else value.stdout.strip()
    return details


def main(argv=None):
    parser = data_adapter.arguments('HW2 BUY/SELL fixed-candidate LLM ranker', model_default=MODEL)
    parser.set_defaults(on_invalid='fallback')
    parser.add_argument('--baseline-dir', type=Path, default=BASELINE)
    parser.add_argument('--cuda-dtype', choices=('bfloat16', 'float16'), default='bfloat16',
                        help='CUDA precision only; use float16 on a Colab T4')
    parser.add_argument('--history-mode', choices=('buy-sell', 'buy-only'), default='buy-sell')
    jobs = parser.add_mutually_exclusive_group(required=True)
    jobs.add_argument('--prepare-only', action='store_true', help='Export all 100 prompts; no inference')
    jobs.add_argument('--smoke', action='store_true', help='One real inference; no evaluation scores')
    jobs.add_argument('--full', action='store_true', help='Explicitly request the 100-user evaluation')
    args = parser.parse_args(argv)
    if args.model == MODEL and not args.revision:
        args.revision = REVISION
    args.history_events = 15
    args.shuffle_seed = None

    destination = args.output_dir.resolve()
    protected = args.baseline_dir.resolve()
    for path in {protected, BASELINE.resolve()}:
        if destination == path or destination.is_relative_to(path):
            parser.error('Refusing to write into the saved baseline directory')
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        parser.error('Use a fresh output directory; prior run artifacts will not be overwritten')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    run = {'status': 'starting', 'mode': 'smoke' if args.smoke else
           'prepare_only' if args.prepare_only else 'full', 'history_mode': args.history_mode,
           'model': args.model, 'requested_revision': args.revision,
           'requested_device': args.device, 'requested_cuda_dtype': args.cuda_dtype,
           'allow_download': args.allow_download,
           'generation': {'max_new_tokens': 512, 'context_limit': 8192,
                          'do_sample': False, 'thinking': False, 'trust_remote_code': False},
           'history_events_per_operation': args.history_events, 'shuffle_seed': args.shuffle_seed,
           'on_invalid': args.on_invalid, 'environment': environment()}
    run_path = args.output_dir / 'experiment.json'
    write_json(run_path, run)
    try:
        bundle = data_adapter.load(args)
        freeze_candidates(bundle, protected)
        if args.smoke:
            bundle, run['smoke_user_index_zero_based'] = smoke_bundle(bundle)
        run['protocol'] = bundle[-1]
        buy_sell = args.history_mode == 'buy-sell'
        kwargs = {'prompt_builder': build_prompt if buy_sell else ranker.build_prompt,
                  'method': 'llmrank_buy_sell' if buy_sell else 'llmrank_buy_only',
                  'adaptation': 'separate_recent_sell_history' if buy_sell else 'sequential_recency_only'}
        ranker.run(Namespace(**dict(vars(args), prepare_only=True)), bundle, **kwargs)
        if args.prepare_only:
            run['status'] = 'prepared_no_inference'
        else:
            run['runtime'] = ranker.run(args, bundle, score_results=not args.smoke, **kwargs)
            valid_smoke = not run['runtime']['raw_invalid_count'] and run['runtime']['new_calls'] == 1
            run['status'] = ('valid_real_inference' if valid_smoke else 'smoke_did_not_pass'
                             ) if args.smoke else 'evaluated'
    except Exception as exc:
        run.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        run['wall_seconds'] = time.perf_counter() - started
        write_json(run_path, run)
    print(f"{run['mode']}: {run['status']} ({run['wall_seconds']:.2f} s)")
    return int(run['status'] == 'smoke_did_not_pass')


if __name__ == '__main__':
    raise SystemExit(main())
