#!/usr/bin/env python3
"""
Voyage batch embedder for scouting reports
- Reads from reports.csv
- Writes to embeddings.csv and embeddings.json
- Auto-detects report column (case-insensitive)
- Skips empty/whitespace rows (keeps row alignment)
- Batches requests with simple retries
"""

import os
import json
import time
from typing import List

import pandas as pd
from tqdm import tqdm
import voyageai

REPORT_COL_CANDIDATES = ["report", "Report", "scouting_report", "Scouting Report", "ScoutingReport"]

def find_report_col(df: pd.DataFrame) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for c in REPORT_COL_CANDIDATES:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    for c in df.columns:
        if "report" in c.lower():
            return c
    raise AssertionError(f"CSV must contain a scouting report text column. Found columns: {list(df.columns)}")

def batched(seq: List[str], n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]

def embed_batches(vo: voyageai.Client, texts: List[str], model: str, batch_size: int, sleep: float):
    out = []
    for batch in tqdm(list(batched(texts, batch_size)), desc=f"Embedding with {model}"):
        for attempt in range(4):
            try:
                resp = vo.embed(texts=batch, model=model, input_type="document")
                out.extend(resp.embeddings)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(sleep * (attempt + 1))
    return out

def main():
    api_key = os.getenv("VOYAGE_API_KEY")
    assert api_key, "❌ VOYAGE_API_KEY not set. Run: export VOYAGE_API_KEY=your_key_here"

    vo = voyageai.Client(api_key=api_key)

    df = pd.read_csv("reports.csv")
    report_col = find_report_col(df)

    rep = df[report_col].astype(str)
    rep = rep.replace({"NA": "", "N/A": "", "—": "", "-": ""}, regex=False)

    nonempty_mask = rep.notna() & rep.str.strip().ne("")
    idx_nonempty = df.index[nonempty_mask]

    texts = rep.loc[idx_nonempty].str.strip().tolist()
    print(f"Found {len(df)} rows; embedding {len(texts)} non-empty reports (skipping {len(df) - len(texts)} empties).")

    embeddings = embed_batches(vo, texts, "voyage-3-large", 100, 1.25)

    assert len(embeddings) == len(idx_nonempty), (
        f"Mismatch: got {len(embeddings)} embeddings for {len(idx_nonempty)} non-empty rows"
    )

    df["embedding"] = None
    emb_series = pd.Series([list(map(float, vec)) for vec in embeddings], index=idx_nonempty, dtype=object)
    df.loc[idx_nonempty, "embedding"] = emb_series

    df.to_csv("embeddings.csv", index=False)
    with open("embeddings.json", "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    print("✅ Done: embeddings.csv and embeddings.json saved.")

if __name__ == "__main__":
    main()
