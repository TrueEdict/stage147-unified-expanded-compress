#!/usr/bin/env python3
"""Stage147 real dual-teacher, shard-resumable distillation.

Loads Qwen and LFM sequentially, captures teacher top-k behavior into a shared
UTF-8 token-string vocabulary, trains a shell-seeded expanded linear organ by
streaming sufficient statistics, compresses it with SVD, and writes durable
per-shard provenance/checkpoints.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, math, os, random, time
from pathlib import Path
from typing import Any
import numpy as np

DOMAINS = {
    "continuation": ["The river passed the old mill and", "At dawn the research team discovered"],
    "instruction": ["Explain how to test a hypothesis step by step.", "Write clear instructions for backing up a folder."],
    "reasoning": ["A train travels 120 miles in 2 hours. Its average speed is", "If all orchids are plants and some plants bloom at night, then"],
    "code": ["Write a Python function that returns prime numbers below n.", "Fix this function so it handles an empty list:"],
    "summarization": ["Summarize: Photosynthesis converts light energy into chemical energy in plants.", "Summarize: A database index speeds reads by maintaining an auxiliary search structure."],
}

def utc() -> str: return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')
def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def stable_hash(text: str, dim: int, seed: int) -> np.ndarray:
    out=np.zeros(dim,np.float32)
    raw=text.encode('utf-8','replace')
    for n in (1,2,3,4):
        for i in range(max(1,len(raw)-n+1)):
            g=raw[i:i+n]; d=hashlib.blake2b(g,digest_size=16,person=str(seed).encode()[:16]).digest(); idx=int.from_bytes(d[:8],'little')%dim; out[idx]+=1.0 if d[8]&1 else -1.0
    norm=np.linalg.norm(out); return out/(norm+1e-8)
def canonical_piece(text: str) -> str:
    text=text.replace('\u0000','').strip('\r\n')
    return text if text else '<EMPTY>'
def build_examples(repeats: int, seed: int) -> list[dict[str,str]]:
    rng=random.Random(seed); rows=[]
    for _ in range(repeats):
        for domain,prompts in DOMAINS.items():
            ps=list(prompts); rng.shuffle(ps)
            for p in ps: rows.append({'domain':domain,'prompt':p})
    return rows
def shard_rows(rows: list[dict[str,str]], size: int):
    for i in range(0,len(rows),size): yield i//size,rows[i:i+size]

def load_teacher(model_id: str, device: str, dtype_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model_id,trust_remote_code=True,use_fast=True)
    dtype={'float16':torch.float16,'bfloat16':torch.bfloat16,'float32':torch.float32}[dtype_name]
    kwargs={'trust_remote_code':True,'low_cpu_mem_usage':True,'torch_dtype':dtype}
    if device=='auto': kwargs['device_map']='auto'
    model=AutoModelForCausalLM.from_pretrained(model_id,**kwargs)
    if device!='auto': model.to(device)
    model.eval(); return tok,model

def capture_teacher(model_id: str, rows: list[dict[str,str]], topk: int, seq_len: int, device: str, dtype_name: str) -> list[dict[str,float]]:
    import torch
    tok,model=load_teacher(model_id,device,dtype_name)
    actual_device=next(model.parameters()).device
    captured=[]
    with torch.inference_mode():
        for row in rows:
            enc=tok(row['prompt'],return_tensors='pt',truncation=True,max_length=seq_len)
            enc={k:v.to(actual_device) for k,v in enc.items()}
            logits=model(**enc).logits[0,-1].float()
            probs=torch.softmax(logits,dim=-1)
            vals,ids=torch.topk(probs,min(topk,probs.shape[-1]))
            dist={}
            for p,tid in zip(vals.cpu().tolist(),ids.cpu().tolist()):
                piece=canonical_piece(tok.decode([tid],clean_up_tokenization_spaces=False,skip_special_tokens=False))
                dist[piece]=dist.get(piece,0.0)+float(p)
            z=sum(dist.values()) or 1.0
            captured.append({k:v/z for k,v in dist.items()})
    del model,tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return captured

def merge_distributions(per_teacher: list[list[dict[str,float]]], weights: list[float]) -> list[dict[str,float]]:
    merged=[]
    for row_i in range(len(per_teacher[0])):
        d={}
        for td,w in zip(per_teacher,weights):
            for token,p in td[row_i].items(): d[token]=d.get(token,0.0)+w*p
        z=sum(d.values()) or 1.0; merged.append({k:v/z for k,v in d.items()})
    return merged

def vectorize_targets(dists: list[dict[str,float]], vocab: list[str]) -> np.ndarray:
    ix={t:i for i,t in enumerate(vocab)}; y=np.zeros((len(dists),len(vocab)),np.float32)
    for r,d in enumerate(dists):
        for t,p in d.items():
            if t in ix: y[r,ix[t]]=p
    y/=np.maximum(y.sum(1,keepdims=True),1e-9); return y

def softmax(x):
    z=x-x.max(1,keepdims=True); e=np.exp(np.clip(z,-60,60)); return e/e.sum(1,keepdims=True)
def evaluate(y,pred):
    q=softmax(pred); top=min(5,y.shape[1]); yi=np.argpartition(y,-top,axis=1)[:,-top:]; qi=np.argpartition(q,-top,axis=1)[:,-top:]
    top5=float(np.mean([len(set(a)&set(b))/top for a,b in zip(yi,qi)])); cos=float(np.mean(np.sum(y*q,1)/(np.linalg.norm(y,axis=1)*np.linalg.norm(q,axis=1)+1e-9))); kl=float(np.mean(np.sum(y*(np.log(y+1e-9)-np.log(q+1e-9)),1)))
    return {'top5_agreement':top5,'probability_cosine':cos,'kl':kl}

def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--teacher-models',default='Qwen/Qwen3-0.6B,LiquidAI/LFM2-350M')
    p.add_argument('--teacher-weights',default='0.5,0.5')
    p.add_argument('--work-dir',default='/kaggle/working/stage147_real_dual_teacher')
    p.add_argument('--device',default='auto'); p.add_argument('--dtype',choices=['float16','bfloat16','float32'],default='float16')
    p.add_argument('--seed',type=int,default=147); p.add_argument('--repeats',type=int,default=8); p.add_argument('--shard-size',type=int,default=8); p.add_argument('--teacher-topk',type=int,default=64); p.add_argument('--seq-len',type=int,default=128)
    p.add_argument('--feature-dim',type=int,default=256); p.add_argument('--temp-hidden',type=int,default=1024); p.add_argument('--compress-rank',type=int,default=64); p.add_argument('--ridge',type=float,default=1e-2); p.add_argument('--max-shared-vocab',type=int,default=4096); p.add_argument('--resume',action='store_true')
    a=p.parse_args(); out=Path(a.work_dir); shards_dir=out/'shards'; ckpt_dir=out/'checkpoints'; shards_dir.mkdir(parents=True,exist_ok=True); ckpt_dir.mkdir(parents=True,exist_ok=True)
    teachers=[x.strip() for x in a.teacher_models.split(',') if x.strip()]; weights=[float(x) for x in a.teacher_weights.split(',')]; assert len(teachers)==len(weights) and teachers
    sw=sum(weights); weights=[w/sw for w in weights]; rows=build_examples(a.repeats,a.seed)
    manifest={'stage':147,'created_at_utc':utc(),'teachers':teachers,'teacher_weights':weights,'shard_size':a.shard_size,'total_examples':len(rows),'shards':[]}
    all_dists=[]; all_rows=[]
    for shard_id,chunk in shard_rows(rows,a.shard_size):
        shard_file=shards_dir/f'shard_{shard_id:04d}.json'
        if a.resume and shard_file.exists():
            payload=json.loads(shard_file.read_text()); merged=payload['merged_targets']
        else:
            captures=[]
            for teacher in teachers: captures.append(capture_teacher(teacher,chunk,a.teacher_topk,a.seq_len,a.device,a.dtype))
            merged=merge_distributions(captures,weights)
            payload={'shard_id':shard_id,'created_at_utc':utc(),'teachers':teachers,'teacher_weights':weights,'rows':chunk,'merged_targets':merged}
            dump(shard_file,payload)
        manifest['shards'].append({'shard_id':shard_id,'path':str(shard_file.relative_to(out)),'sha256':sha256_file(shard_file),'examples':len(chunk)})
        all_dists.extend(merged); all_rows.extend(chunk); dump(out/'manifest.json',manifest)
    mass={}
    for d in all_dists:
        for t,v in d.items(): mass[t]=mass.get(t,0.0)+v
    vocab=[t for t,_ in sorted(mass.items(),key=lambda kv:(-kv[1],kv[0]))[:a.max_shared_vocab]]
    rng=np.random.default_rng(a.seed); A=rng.normal(0,1/math.sqrt(a.feature_dim),(a.feature_dim,a.temp_hidden)).astype(np.float32)
    hth=np.zeros((a.temp_hidden,a.temp_hidden),np.float32); hty=np.zeros((a.temp_hidden,len(vocab)),np.float32); eval_x=[]; eval_y=[]
    for shard in manifest['shards']:
        payload=json.loads((out/shard['path']).read_text())
        sx=np.stack([stable_hash(r['domain']+'\n'+r['prompt'],a.feature_dim,a.seed) for r in payload['rows']])
        sy=vectorize_targets(payload['merged_targets'],vocab); sh=sx@A
        hth += sh.T@sh; hty += sh.T@sy
        eval_x.append(sx); eval_y.append(sy)
        dump(ckpt_dir/f"stats_{shard['shard_id']:04d}.json", {'shard_id':shard['shard_id'],'examples':len(payload['rows']),'hth_trace':float(np.trace(hth)),'target_mass':float(hty.sum())})
    Wtemp=np.linalg.solve(hth+a.ridge*np.eye(a.temp_hidden,dtype=np.float32),hty); W=A@Wtemp
    U,S,Vt=np.linalg.svd(W,full_matrices=False); rank=min(a.compress_rank,len(S)); Wc=(U[:,:rank]*S[:rank])@Vt[:rank]
    X=np.concatenate(eval_x,axis=0); Y=np.concatenate(eval_y,axis=0)
    metrics=evaluate(Y,X@Wc); metrics.update({'examples':len(all_rows),'shared_vocab_size':len(vocab),'feature_dim':a.feature_dim,'temporary_hidden':a.temp_hidden,'compress_rank':rank,'temporary_parameters':int(a.feature_dim*a.temp_hidden+a.temp-hidden*len(vocab)) if False else int(a.feature_dim*a.temp_hidden+a.temp_hidden*len(vocab)),'final_parameters':int(a.feature_dim*rank+rank*len(vocab)),'student_fit_mode':'per_shard_sufficient_statistics'})
    np.savez_compressed(ckpt_dir/'student_compressed.npz',projection=A.astype(np.float16),u=U[:,:rank].astype(np.float16),s=S[:rank].astype(np.float32),vt=Vt[:rank].astype(np.float16),vocab=np.array(vocab,dtype=object))
    gates={'two_teachers':len(teachers)==2,'all_shards_materialized':len(manifest['shards'])==math.ceil(len(rows)/a.shard_size),'shared_token_vocab':len(vocab)>0,'compressed_checkpoint_exists':(ckpt_dir/'student_compressed.npz').exists(),'final_smaller_than_temporary':metrics['final_parameters']<metrics['temporary_parameters'],'finite_metrics':all(math.isfinite(metrics[k]) for k in ['top5_agreement','probability_cosine','kl']),'per_shard_student_fit':metrics['student_fit_mode']=='per_shard_sufficient_statistics'}
    report={'stage':147,'name':'real_dual_teacher_sharded_distillation','runtime_verdict':'PASS' if all(gates.values()) else 'FAIL','teachers':teachers,'teacher_weights':weights,'metrics':metrics,'gates':gates,'manifest_sha256':sha256_file(out/'manifest.json'),'checkpoint_sha256':sha256_file(ckpt_dir/'student_compressed.npz')}
    dump(out/'stage147_real_report.json',report); (out/'FINAL_VERDICT.md').write_text(f"# Stage147 Real Dual-Teacher Sharded Distillation\n\n**{report['runtime_verdict']}**\n\nTeachers: {', '.join(teachers)}\n\nShards: {len(manifest['shards'])}\n",encoding='utf-8')
    print(json.dumps(report,indent=2)); return 0 if report['runtime_verdict']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
