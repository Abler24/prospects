#!/usr/bin/env python3
# print_top_earners.py — check that we're pulling earnings from fusion files

import os, json, ast
import numpy as np

FILES = [
    "fusion/trainScouting.jsonl",
    "fusion/dev.jsonl",
    "fusion/testScouting.jsonl",
]
EARNINGS_KEYS = ["Total_Earnings","total_earnings","TotalEarnings",
                 "earnings_total","CareerEarnings","career_earnings", "salary"]
EMBED_DIM = 1024

def read_jsonl(path):
    with open(path,"r",encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            s=line.strip()
            if not s: continue
            try: yield i,json.loads(s)
            except: yield i,ast.literal_eval(s)

def parse_num(x):
    if x is None: return 0.0
    if isinstance(x,(int,float)): return float(x)
    s=str(x).replace(",","").replace("$","").strip()
    try: return float(s)
    except: return 0.0

def get_earnings(row):
    for k in EARNINGS_KEYS:
        if k in row: return parse_num(row[k])
    stats=row.get("stats",{})
    if isinstance(stats,dict):
        for k in EARNINGS_KEYS:
            if k in stats: return parse_num(stats[k])
    return 0.0

def get_name(row):
    for k in ["Name","name","player","player_name","name_key"]:
        if k in row and str(row[k]).strip():
            return str(row[k]).strip()
    stats=row.get("stats",{})
    if isinstance(stats,dict):
        for k in ["Name","name","player","player_name","name_key"]:
            if k in stats and str(stats[k]).strip():
                return str(stats[k]).strip()
    return "Unknown"

def has_embedding(row):
    if "embedding" not in row: return False
    val=row["embedding"]
    if isinstance(val,(list,tuple,np.ndarray)):
        return len(val)==EMBED_DIM
    if isinstance(val,str) and val.startswith("["):
        try:
            arr=np.array(json.loads(val))
            return arr.ndim==1 and arr.shape[0]==EMBED_DIM
        except: return False
    return False

def main():
    for path in FILES:
        if not os.path.exists(path):
            print(f"[MISS] {path}")
            continue
        rows=list(read_jsonl(path))
        earners=[]
        for i,row in rows:
            earn=get_earnings(row)
            nm=get_name(row)
            emb=has_embedding(row)
            earners.append((earn,nm,emb,i))
        earners.sort(reverse=True,key=lambda x:x[0])
        print("\n"+"="*60)
        print(f"FILE: {path} (total {len(rows)} rows)")
        for earn,nm,emb,lineno in earners[:15]:
            print(f"{nm:<24} ${earn:>12,.0f} | Embedding? {emb} | line {lineno}")

if __name__=="__main__":
    main()
