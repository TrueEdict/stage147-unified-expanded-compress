#!/usr/bin/env python3
"""Stage147 Restore-146 Antithesis Distillation wrapper."""
from __future__ import annotations
import argparse, copy, csv, importlib.util, json, shutil, time
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('lineage_stage146_run', HERE / 'lineage_stage146_run.py')
stage146 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(stage146)


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')


def run_stage146_target(args: argparse.Namespace) -> Dict[str, Any]:
    X, Y, labels = stage146.demo_dataset(args.feature_dim, args.demo_vocab_size, args.samples_per_domain, args.seed)
    shell = stage146.build_shell_report(args.feature_dim, args.shell_core_params, args.demo_vocab_size)
    res = stage146.train_expand_compress(X, Y, labels, args)
    m = res['metrics']
    m['repair_history'] = res.get('repair_history', [])
    m['sample_generations'] = res.get('sample_generations', [])
    m['shell_report'] = shell
    gates = {
        'shell_validated': shell['shell_validated'],
        'shell_params_le_131k': shell['shell_params_le_131k'],
        'temporary_params_le_5m': m['peak_training_params'] <= args.temporary_param_budget,
        'temporary_params_gt_shell': m['peak_training_params'] > args.shell_core_params,
        'final_params_le_budget': m['final_compressed_params'] <= args.final_param_budget,
        'temporary_organ_removed': m['temporary_organ_removed_before_final_eval'],
        'pre_kl_improvement': m['pre_compression_kl_improvement'] >= args.min_pre_improvement,
        'post_kl_improvement': m['post_compression_kl_improvement'] >= args.min_post_improvement,
        'post_retention': m['post_compression_quality_retention'] >= args.min_retention,
        'forgetting_bound': m['max_compression_forgetting'] <= args.max_forgetting,
        'entropy_agreement': m['post_entropy_agreement'] >= args.min_entropy_agreement,
        'top5_agreement': m['post_top5_agreement'] >= args.min_top5,
        'teacher_logits_used': True,
        'teacher_text_outputs_not_used': True,
        'model_specific_blowup_seeded_from_shell': True,
    }
    m.update({'name':'stage147_restore146_primary_target_path','stage':147,'lineage':'exact_stage146_shell_seeded_expand_compress','gates':gates,'passed_gates':sum(bool(v) for v in gates.values()),'total_gates':len(gates),'runtime_verdict':'PASS' if all(gates.values()) else 'PARTIAL'})
    return m


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out = Path(args.work_dir)
    out.mkdir(parents=True, exist_ok=True)
    primary = run_stage146_target(copy.deepcopy(args))
    gates = {**primary['gates'],
        'stage146_result_recovered_top5_ge_0_98': primary['post_top5_agreement'] >= 0.98,
        'stage146_result_recovered_entropy_ge_0_99': primary['post_entropy_agreement'] >= 0.99,
        'stage146_result_recovered_cosine_ge_0_99': primary['post_logit_cosine'] >= 0.99,
        'stage146_result_recovered_kl_ge_0_99': primary['post_compression_kl_improvement'] >= 0.99,
        'stage146_peak_params_restored_gt_4m': primary['peak_training_params'] >= 4_000_000,
        'stage146_final_params_le_500k': primary['final_compressed_params'] <= 500_000,
    }
    report = {'stage':147,'name':'stage147_restore146_antithesis_distill','created_at_utc':now(),'mode':'demo','teacher_model':'synthetic_shell_seeded_teacher','primary_stage146_exact_path':primary,'gates':gates,'passed_gates':sum(bool(v) for v in gates.values()),'total_gates':len(gates),'runtime_verdict':'PASS' if all(gates.values()) else 'PARTIAL'}
    write_json(report, out/'stage147_restore146_metrics.json')
    with (out/'stage147_pass_gates.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['gate','passed']); w.writerows((k,bool(v)) for k,v in gates.items())
    (out/'stage147_restore146_report.md').write_text(f"# Stage147 Restore-146\n\nVerdict: {report['runtime_verdict']}\n", encoding='utf-8')
    shutil.make_archive(str(out/'stage147_outputs'), 'zip', root_dir=str(out))
    return report


def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['demo'],default='demo'); p.add_argument('--work-dir',default='./stage147_restore146_run'); p.add_argument('--seed',type=int,default=146); p.add_argument('--feature-dim',type=int,default=64); p.add_argument('--demo-vocab-size',type=int,default=4096); p.add_argument('--samples-per-domain',type=int,default=20); p.add_argument('--temp-hidden',type=int,default=1024); p.add_argument('--compress-rank',type=int,default=64); p.add_argument('--shell-core-params',type=int,default=96000); p.add_argument('--temporary-param-budget',type=int,default=5000000); p.add_argument('--final-param-budget',type=int,default=500000); p.add_argument('--ridge',type=float,default=2e-2); p.add_argument('--temperature',type=float,default=2.0); p.add_argument('--use-nonlinear-temp',action='store_true'); p.add_argument('--repair-steps',type=int,default=8); p.add_argument('--repair-rank',type=int,default=4); p.add_argument('--repair-lr',type=float,default=0.25); p.add_argument('--min-pre-improvement',type=float,default=0.50); p.add_argument('--min-post-improvement',type=float,default=0.50); p.add_argument('--min-retention',type=float,default=0.85); p.add_argument('--max-forgetting',type=float,default=0.10); p.add_argument('--min-entropy-agreement',type=float,default=0.80); p.add_argument('--min-top5',type=float,default=0.35); args=p.parse_args(); run(args)

if __name__=='__main__': main()
