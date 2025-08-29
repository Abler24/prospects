#!/usr/bin/env python3
# server.py — 4-class, ordered Stats block, persistent selected name display, and MLB comp by embedding
# CHANGES REQUESTED (2025-08-29):
# - Tilt bucket probabilities in the displayed bars to be more favored toward higher buckets (top bucket 1.4× vs lowest).
# - Predicted earnings (EV) multiplied by 1.4× for display.
# - Change the <$1,000,000 display rule: only show it when predicted bucket == 1 AND class-1 combined prob > 0.40; otherwise show EV.
# - Keep the underlying model logits/probabilities for logic; only the UI bar displays are tilted.

import os, ast, json, math, re
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from torch import nn
from flask import Flask, request, render_template_string

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
EMBED_DIM = 1024

# ── Paths ────────────────────────────────────────────────────────────────────
PROSPECTS_PATH = "cleaned_prospects.jsonl"

# 2-class (binary MLB) models
STATS_P_CKPT_2 = "checkpoints/statsP.pt"
STATS_H_CKPT_2 = "checkpoints/statsH.pt"
SCOUT_CKPT_2   = "checkpoints/scout.pt"

# 4-class models
STATS_P_CKPT_4 = "checkpoints_4/statsP.pt"
STATS_H_CKPT_4 = "checkpoints_4/statsH.pt"
SCOUT_CKPT_4   = "checkpoints_4/scout.pt"

# Optional temperatures
TEMPS_4_JSON = "checkpoints_4/temps.json"
TEMPS_4_PT   = "checkpoints_4/temps.pt"
TEMPS_2_JSON = "checkpoints/temps.json"
TEMPS_2_PT   = "checkpoints/temps.pt"

# Salary bucket cutoffs (defaults: 1M / 5M / 50M)
C1 = float(os.getenv("CUT1", "1000000"))      # 1↔2
C2 = float(os.getenv("CUT2", "5000000"))      # 2↔3
C3 = float(os.getenv("CUT3", "50000000"))     # 3↔4
LAST_BUCKET_CAP = float(os.getenv("LAST_BUCKET_CAP", "150000000"))

BUCKET_LOWER = np.array([0.0, C1, C2, C3], dtype=np.float64)
BUCKET_UPPER = np.array([C1, C2, C3, LAST_BUCKET_CAP], dtype=np.float64)
BUCKET_MIDS  = np.array([
    0.5 * (BUCKET_LOWER[0] + BUCKET_UPPER[0]),
    0.5 * (BUCKET_LOWER[1] + BUCKET_UPPER[1]),
    0.5 * (BUCKET_LOWER[2] + BUCKET_UPPER[2]),
    0.5 * (BUCKET_LOWER[3] + BUCKET_UPPER[3]),
], dtype=np.float64)

# Filters & binary threshold
MIN_G_HITTER = 100
MIN_IP_PITCHER = 70.0
BINARY_THRESH = float(os.environ.get("BINARY_THRESH", "0.5"))

# Visual “spice” for scouting bars (display-only)
SCOUT_VISUAL_GAMMA = float(os.environ.get("SCOUT_VISUAL_GAMMA", "1.35"))
SCOUT_VISUAL_BLEND = float(os.environ.get("SCOUT_VISUAL_BLEND", "0.50"))
SCOUT_VISUAL_MIN_LIFT = float(os.environ.get("SCOUT_VISUAL_MIN_LIFT", "1e-6"))

# Default scout temps
DEFAULT_SCOUT_T2 = float(os.environ.get("SCOUT_T2_DEFAULT", "0.85"))
DEFAULT_SCOUT_T4 = float(os.environ.get("SCOUT_T4_DEFAULT", "0.85"))

# NEW: Tilt and EV scalar controls (as requested)
TILT_HIGH_FACTOR = float(os.environ.get("TILT_HIGH_FACTOR", "1.4"))  # top bucket is 1.4× vs bottom
EV_DISPLAY_SCALAR = float(os.environ.get("EV_DISPLAY_SCALAR", "1.6"))  # multiply EV by 1.4× for display

# ── Rate stats keys ─────────────────────────────────────────────────────────
HIT_RATE_STATS = [
    "AVG","BB%","K%","BB/K","OBP","SLG","OPS","ISO","BABIP","Spd","wSB","wRC",
    "wRAA","wOBA","wRC+","GB/FB","LD%","GB%","FB%","IFFB%","HR/FB","Pull%","Cent%",
    "Oppo%","SwStr%"
]
PITCH_RATE_STATS = [
    "ERA","K/9","BB/9","K/BB","HR/9","K%","BB%","K-BB%","AVG","WHIP","BABIP","LOB%",
    "ERA_y","FIP","E-F","xFIP","GB/FB","LD%","GB%","FB%","IFFB%","HR/FB","Pull%",
    "Cent%","Oppo%","SwStr%"
]

# ── Model ────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden=(512, 256), p_drop=0.05):
        super().__init__()
        layers = []
        dims = [in_dim] + list(hidden)
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(p_drop)]
        layers += [nn.Linear(dims[-1], out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

_layer_idx_re = re.compile(r"net\.(\d+)\.(weight|bias)$")

def _infer_dims_from_state_dict(sd: dict):
    pairs = []
    for k, v in sd.items():
        if k.endswith('.weight') and isinstance(v, torch.Tensor) and v.ndim == 2:
            m = _layer_idx_re.search(k)
            if m: pairs.append((int(m.group(1)), v))
    if not pairs:
        mats = [v for v in sd.values() if isinstance(v, torch.Tensor) and v.ndim == 2]
        if not mats: raise ValueError("Cannot infer dimensions")
        out_dim = int(min(w.shape[0] for w in mats))
        in_dim  = int(max(w.shape[1] for w in mats))
        return in_dim, out_dim
    pairs.sort(key=lambda x: x[0])
    first_w = pairs[0][1]; last_w  = pairs[-1][1]
    return int(first_w.shape[1]), int(last_w.shape[0])

def load_stats_ckpt(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = ckpt.get("model_state", ckpt.get("state_dict"))
    if sd is None and isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        sd, ckpt = ckpt, {}
    in_dim = int(ckpt.get("in_dim", _infer_dims_from_state_dict(sd)[0]))
    out_dim = int(ckpt.get("out_dim", _infer_dims_from_state_dict(sd)[1]))
    model = MLP(in_dim=in_dim, out_dim=out_dim).to(DEVICE)
    model.load_state_dict(sd); model.eval()
    feat_names = ckpt.get("feature_names", ckpt.get("features"))
    if feat_names is None: raise KeyError(f"{path}: missing feature_names")
    mean = np.array(ckpt.get("scaler_mean"), dtype=np.float32)
    std  = np.array(ckpt.get("scaler_std"), dtype=np.float32)
    std[std == 0] = 1.0
    return model, feat_names, mean, std, out_dim

def load_scout_ckpt_embedding_only(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = ckpt.get("model_state", ckpt.get("state_dict"))
    if sd is None and isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        sd, ckpt = ckpt, {}
    in_dim = int(ckpt.get("in_dim", _infer_dims_from_state_dict(sd)[0]))
    out_dim = int(ckpt.get("out_dim", _infer_dims_from_state_dict(sd)[1]))
    model = MLP(in_dim=in_dim, out_dim=out_dim).to(DEVICE)
    model.load_state_dict(sd); model.eval()
    emb_mean = np.array(ckpt.get("embed_mean", None), dtype=np.float32) if "embed_mean" in ckpt else None
    emb_std  = np.array(ckpt.get("embed_std",  None), dtype=np.float32) if "embed_std"  in ckpt else None
    if emb_std is not None: emb_std[emb_std == 0] = 1.0
    return model, emb_mean, emb_std, in_dim, out_dim

# ── Temperatures ─────────────────────────────────────────────────────────────

def _load_json(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {k: float(v) for k, v in d.items()}

def _load_pt(path: str) -> Dict[str, float]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for k, v in d.items():
        out[k] = float(v.item() if isinstance(v, torch.Tensor) else v)
    return out

def load_temperatures() -> Dict[str, float]:
    temps: Dict[str, float] = {}
    for p in [TEMPS_4_JSON, TEMPS_4_PT, TEMPS_2_JSON, TEMPS_2_PT]:
        if not os.path.exists(p): continue
        try: temps.update(_load_json(p) if p.endswith(".json") else _load_pt(p))
        except Exception: pass
    return temps

TEMPS = load_temperatures()


def get_temp(temps: Dict[str, float], key: str, fallback_keys: List[str], default: float = 1.0) -> float:
    for k in [key] + fallback_keys:
        if k in temps:
            return float(temps[k])
    return float(default)


def apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    return logits if T == 1.0 else logits / T

# ── Canonicalization & helpers ───────────────────────────────────────────────

def canon(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace('%',' pct ').replace('/',' per ').replace('+',' plus ').replace('-',' ')
    s = re.sub(r'\s+',' ',s).strip()
    s = re.sub(r'[^a-z0-9]+','',s)
    return s


def parse_num(x: Any) -> float:
    if isinstance(x, (int, float, np.integer, np.floating)):
        return 0.0 if (isinstance(x, float) and math.isnan(x)) else float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}: return 0.0
    if s.endswith("%"):
        try: return float(s[:-1].replace(",", "")) / 100.0
        except: return 0.0
    try: return float(s.replace(",", ""))
    except: return 0.0


def build_canon_row(row: Dict[str, Any]) -> Dict[str, float]:
    stats = row.get("stats", row)
    out = {}
    if isinstance(stats, dict):
        for k, v in stats.items():
            ck = canon(k)
            if ck: out[ck] = parse_num(v)
    return out


def vector_from_row_canon(row_canon: Dict[str, float], feat_names: List[str]) -> np.ndarray:
    return np.array([row_canon.get(canon(f), 0.0) for f in feat_names], dtype=np.float32)


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def parse_embedding(val, emb_mean=None, emb_std=None):
    if val is None: raise ValueError("Missing embedding")
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.array(val, dtype=np.float32)
    elif isinstance(val, str):
        s = val.strip()
        try: arr = np.array(json.loads(s), dtype=np.float32)
        except Exception: arr = np.array(ast.literal_eval(s), dtype=np.float32)
    else:
        raise ValueError("Bad embedding type")
    if emb_mean is not None and emb_std is not None:
        arr = (arr - emb_mean) / emb_std
    if arr.shape[0] != EMBED_DIM:
        raise ValueError(f"Embedding dim mismatch: got {arr.shape[0]}, expected {EMBED_DIM}")
    return arr


def route_is_hitter(row: Dict[str, Any]) -> bool:
    stats = row.get("stats", row)
    if not isinstance(stats, dict): return False
    return any(k.upper() == "OBP" for k in stats.keys())


def coverage_nonzero(vec: np.ndarray) -> bool:
    return bool(np.any(vec != 0.0))


def nice_name(s: str) -> str:
    if not s: return s
    parts = re.split(r"\s+", s.strip())
    return " ".join(p[:1].upper() + p[1:].lower() if p else "" for p in parts)

# ── Filtering & rescaling ────────────────────────────────────────────────────

def passes_threshold(row: Dict[str, Any]) -> bool:
    rc = build_canon_row(row)
    if route_is_hitter(row):
        return rc.get("g", 0.0) >= MIN_G_HITTER
    else:
        return rc.get("ip", 0.0) >= MIN_IP_PITCHER


def compute_dataset_stats(rows: List[Dict[str, Any]], is_hitter: bool) -> Dict[str, Tuple[float, float]]:
    keys = HIT_RATE_STATS if is_hitter else PITCH_RATE_STATS
    buckets: Dict[str, List[float]] = {canon(k): [] for k in keys}
    for row in rows:
        stats = row.get("stats", row)
        if not isinstance(stats, dict): continue
        for k, v in stats.items():
            if k in keys:
                buckets[canon(k)].append(parse_num(v))
    out: Dict[str, Tuple[float, float]] = {}
    for ck, vals in buckets.items():
        if len(vals) == 0: continue
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        out[ck] = (mu, sd if sd != 0.0 else 1.0)
    return out


def rescale_rate_stats(row_canon: Dict[str, float], is_hitter: bool,
                       train_stats: Dict[str, Tuple[float, float]],
                       prospect_stats: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    adjusted = dict(row_canon)
    keys = HIT_RATE_STATS if is_hitter else PITCH_RATE_STATS
    for key in keys:
        ck = canon(key)
        if ck in adjusted and ck in train_stats and ck in prospect_stats:
            mu_tr, sd_tr = train_stats[ck]
            mu_pr, sd_pr = prospect_stats[ck]
            if sd_pr != 0.0:
                adjusted[ck] = ((adjusted[ck] - mu_pr) / sd_pr) * sd_tr + mu_tr
    return adjusted

# ── IO ───────────────────────────────────────────────────────────────────────

def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            try: rows.append(json.loads(s))
            except Exception: rows.append(ast.literal_eval(s))
    return rows

# ── App bootstrap ────────────────────────────────────────────────────────────
prospect_rows = read_jsonl(PROSPECTS_PATH)
pros_hit_all = [r for r in prospect_rows if route_is_hitter(r)]
pros_pit_all = [r for r in prospect_rows if not route_is_hitter(r)]
prospect_stats_h = compute_dataset_stats(pros_hit_all, True)
prospect_stats_p = compute_dataset_stats(pros_pit_all, False)

# Binary feature μ/σ maps (only used to build vectors for 2-class models)
modelP2, featP2, meanP2, stdP2, _ = load_stats_ckpt(STATS_P_CKPT_2)
modelH2, featH2, meanH2, stdH2, _ = load_stats_ckpt(STATS_H_CKPT_2)
modelS2, emb_mean2, emb_std2, IN_S2, OUT_S2 = load_scout_ckpt_embedding_only(SCOUT_CKPT_2)

# Four-class feature μ/σ maps
modelP4, featP4, meanP4, stdP4, _ = load_stats_ckpt(STATS_P_CKPT_4)
modelH4, featH4, meanH4, stdH4, _ = load_stats_ckpt(STATS_H_CKPT_4)
modelS4, emb_mean4, emb_std4, IN_S4, OUT_S4 = load_scout_ckpt_embedding_only(SCOUT_CKPT_4)

train_stats_h4 = {canon(f): (m, s) for f, m, s in zip(featH4, meanH4, stdH4)}
train_stats_p4 = {canon(f): (m, s) for f, m, s in zip(featP4, meanP4, stdP4)}
train_stats_h2 = {canon(f): (m, s) for f, m, s in zip(featH2, meanH2, stdH2)}
train_stats_p2 = {canon(f): (m, s) for f, m, s in zip(featP2, meanP2, stdP2)}


def build_name_list():
    kept = []
    index_by_name = {}
    for i, row in enumerate(prospect_rows):
        if not passes_threshold(row): continue
        emb = row.get("embedding", None)
        if not (isinstance(emb, (list, tuple)) and len(emb) > 0): continue
        rc = build_canon_row(row)
        if route_is_hitter(row):
            x = vector_from_row_canon(rescale_rate_stats(rc, True, train_stats_h4, prospect_stats_h), featH4)
        else:
            x = vector_from_row_canon(rescale_rate_stats(rc, False, train_stats_p4, prospect_stats_p), featP4)
        if not coverage_nonzero(x): continue
        name = str(row.get("Name", "")).strip()
        if not name: continue
        local_i = len(kept)
        kept.append(name)
        index_by_name[name] = (local_i, i)
    return sorted(kept, key=lambda s: s.lower()), index_by_name

KEPT_NAMES, INDEX_BY_NAME = build_name_list()

# ── Temps helper ─────────────────────────────────────────────────────────────

def temps_for_role(is_hitter: bool):
    T2_stats = get_temp(TEMPS, "statsH_T_2" if is_hitter else "statsP_T_2", ["statsH_T","statsP_T"], 1.0)
    T2_scout = get_temp(TEMPS, "scout_T_2", ["scout_T"], DEFAULT_SCOUT_T2)
    T4_stats = get_temp(TEMPS, "statsH_T_4" if is_hitter else "statsP_T_4", ["statsH_T","statsP_T"], 1.0)
    T4_scout = get_temp(TEMPS, "scout_T_4", ["scout_T"], DEFAULT_SCOUT_T4)
    return T2_stats, T2_scout, T4_stats, T4_scout

# ── Display transforms (tilt & spice) ────────────────────────────────────────

def spice_probs_for_display(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.sum() <= 0:
        return np.full_like(p, 1.0 / len(p))
    p = p / p.sum()
    p0 = np.clip(p - p.min(), 0.0, None) + SCOUT_VISUAL_MIN_LIFT
    p_spice = np.power(p0, SCOUT_VISUAL_GAMMA)
    p_spice = p_spice / max(p_spice.sum(), 1e-12)
    lam = max(0.0, min(1.0, SCOUT_VISUAL_BLEND))
    p_vis = (1.0 - lam) * p + lam * p_spice
    s = p_vis.sum()
    if s <= 0:
        return np.full_like(p_vis, 1.0 / len(p_vis))
    return (p_vis / s).astype(np.float64)


def tilt_high_buckets(p: np.ndarray, factor_hi: float = TILT_HIGH_FACTOR) -> np.ndarray:
    """Linearly tilt probabilities toward higher buckets for DISPLAY ONLY.
    Lowest bucket weight = 1.0; highest bucket weight = factor_hi. Intermediate buckets linearly interpolated.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.sum() <= 0 or len(p) <= 1:
        return np.full_like(p, 1.0 / max(len(p), 1))
    k = len(p)
    # Linear ramp weights from 1.0 up to factor_hi
    w = np.array([1.0 + (factor_hi - 1.0) * (i / (k - 1)) for i in range(k)], dtype=np.float64)
    q = p * w
    s = q.sum()
    if s <= 0:
        return np.full_like(q, 1.0 / len(q))
    return (q / s).astype(np.float64)

# ── MLB Comp Index (unchanged) ───────────────────────────────────────────────
FUSION_FILES = [
    "fusion/trainScouting.jsonl",
    "fusion/dev.jsonl",
    "fusion/testScouting.jsonl",
]
EARNINGS_KEY_CANDIDATES = ["Total_Earnings", "total_earnings", "TotalEarnings", "earnings_total", "salary"]
EARNINGS_MIN = 10_000_000.0


def _get_earnings(row: Dict[str, Any]) -> float:
    for k in EARNINGS_KEY_CANDIDATES:
        if k in row:
            return parse_num(row.get(k))
    stats = row.get("stats", {})
    if isinstance(stats, dict):
        for k in EARNINGS_KEY_CANDIDATES:
            if k in stats:
                return parse_num(stats.get(k))
    return 0.0


def _get_name_generic(row: Dict[str, Any]) -> str:
    for k in ["Name","name","player","player_name","name_key"]:
        if k in row and str(row[k]).strip():
            return str(row[k]).strip()
    return ""


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def load_mlb_comp_index(emb_mean: np.ndarray, emb_std: np.ndarray):
    comps: List[Dict[str, Any]] = []
    vecs: List[np.ndarray] = []
    for path in FUSION_FILES:
        if not os.path.exists(path): continue
        try:
            rows = read_jsonl(path)
        except Exception:
            continue
        for row in rows:
            if "embedding" not in row: continue
            earnings = _get_earnings(row)
            if earnings <= EARNINGS_MIN:
                continue
            try:
                e = parse_embedding(row["embedding"], emb_mean, emb_std)
            except Exception:
                continue
            e = _unit(e.astype(np.float32))
            name = _get_name_generic(row) or "Unknown"
            comps.append({
                "name": nice_name(name),
                "earnings": float(earnings),
                "source": os.path.basename(path)
            })
            vecs.append(e)
    comp_mat = np.stack(vecs, axis=0) if vecs else np.zeros((0, EMBED_DIM), dtype=np.float32)
    return comps, comp_mat

# build comp index using 4-class scout normalization space
MLB_COMPS_META, MLB_COMPS_MATRIX = load_mlb_comp_index(emb_mean4, emb_std4)


def nearest_comp(emb_4space: np.ndarray):
    if MLB_COMPS_MATRIX.shape[0] == 0:
        return None
    q = _unit(emb_4space.astype(np.float32))
    sims = (MLB_COMPS_MATRIX @ q)  # cosine since unit vectors
    idx = int(np.argmax(sims))
    return {
        "name": MLB_COMPS_META[idx]["name"],
        "similarity": float(sims[idx]),
        "earnings": int(round(MLB_COMPS_META[idx]["earnings"])),
        "source": MLB_COMPS_META[idx]["source"]
    }

# ── Per-player prediction ────────────────────────────────────────────────────

def predict_for_name(name: str):
    tup = INDEX_BY_NAME.get(name)
    if tup is None:
        return {"error": "Name not found in filtered cohort (needs embedding + usable stats)."}

    _, gi = tup
    row = prospect_rows[gi]
    is_h = route_is_hitter(row)
    rc = build_canon_row(row)

    # Rescale to training μ/σ per task
    rc2 = rescale_rate_stats(rc, is_h, train_stats_h2 if is_h else train_stats_p2,
                             prospect_stats_h if is_h else prospect_stats_p)
    rc4 = rescale_rate_stats(rc, is_h, train_stats_h4 if is_h else train_stats_p4,
                             prospect_stats_h if is_h else prospect_stats_p)

    if is_h:
        x2 = vector_from_row_canon(rc2, featH2); x2 = standardize(x2, meanH2, stdH2)
        x4 = vector_from_row_canon(rc4, featH4); x4 = standardize(x4, meanH4, stdH4)
        m2, m4 = modelH2, modelH4
        role = "B"
    else:
        x2 = vector_from_row_canon(rc2, featP2); x2 = standardize(x2, meanP2, stdP2)
        x4 = vector_from_row_canon(rc4, featP4); x4 = standardize(x4, meanP4, stdP4)
        m2, m4 = modelP2, modelP4
        role = "P"

    emb2 = parse_embedding(row.get("embedding"), emb_mean2, emb_std2)
    emb4 = parse_embedding(row.get("embedding"), emb_mean4, emb_std4)

    T2_stats, T2_scout, T4_stats, T4_scout = temps_for_role(is_h)

    with torch.no_grad():
        # Binary (2-class) combined MLB prob
        ls2_stats = apply_temperature(m2(torch.from_numpy(x2).to(DEVICE).unsqueeze(0)), T2_stats)
        ls2_scout = apply_temperature(modelS2(torch.from_numpy(emb2).to(DEVICE).unsqueeze(0)), T2_scout)
        ps2_stats = torch.softmax(ls2_stats, dim=1).cpu().numpy().flatten()
        ps2_scout = torch.softmax(ls2_scout, dim=1).cpu().numpy().flatten()
        ps2_comb  = torch.softmax(ls2_stats + ls2_scout, dim=1).cpu().numpy().flatten()
        mlb_comb = float(ps2_comb[1])
        mlb_yes_no = "Yes" if mlb_comb >= BINARY_THRESH else "No"

        # Four-class (projected earnings)
        ls4_stats = apply_temperature(m4(torch.from_numpy(x4).to(DEVICE).unsqueeze(0)), T4_stats)
        ls4_scout = apply_temperature(modelS4(torch.from_numpy(emb4).to(DEVICE).unsqueeze(0)), T4_scout)
        p4_stats  = torch.softmax(ls4_stats, dim=1).cpu().numpy().flatten()
        p4_scout_raw = torch.softmax(ls4_scout, dim=1).cpu().numpy().flatten()

        # Display-only transforms:
        #  1) Scout: spice then tilt toward higher buckets
        p4_scout_vis = spice_probs_for_display(p4_scout_raw)
        p4_scout_vis = tilt_high_buckets(p4_scout_vis, TILT_HIGH_FACTOR)
        #  2) Stats: tilt only (no spice)
        p4_stats_vis = tilt_high_buckets(p4_stats, TILT_HIGH_FACTOR)

        # Combined (for class selection / EV) — keep original logic
        lc4_comb  = (ls4_stats + ls4_scout).cpu().numpy().flatten()
        p4_comb   = torch.softmax(torch.from_numpy(lc4_comb), dim=0).cpu().numpy()
        cls_idx   = int(np.argmax(lc4_comb))

    # Expected earnings from ALL logits (prob-weighted mids), then × EV scalar, rounded to nearest $1k
    ev_from_logits = float(np.dot(p4_comb, BUCKET_MIDS))
    ev_from_logits *= EV_DISPLAY_SCALAR
    ev_from_logits = int(round(ev_from_logits / 1000.0) * 1000)

    # Display string rule (UPDATED):
    # If predicted class is 1 and combined prob for class-1 > 0.40, show "<$1,000,000"; else show EV.
    ev_display = None
    if cls_idx == 0 and float(p4_comb[0]) > 0.40:
        ev_display = "<$1,000,000"
    else:
        ev_display = f"${ev_from_logits:,.0f}"

    # Scouting report text
    report_text = None
    rpt = row.get("Report")
    if isinstance(rpt, str) and len(rpt.strip()) > 0:
        report_text = rpt.strip()

    # ── Ordered Stats KV (unchanged ordering rules) ───────────────────────────
    kv_source = row.get("stats", row)
    base_name = row.get("Name") or row.get("name_key") or ""
    pretty_name = nice_name(str(base_name)) if base_name else nice_name(name)

    raw_items = {}
    if isinstance(kv_source, dict):
        for k, v in kv_source.items():
            if v is None: continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                s = f"{v:.3f}".rstrip("0").rstrip(".")
            else:
                s = str(v)
            raw_items[str(k)] = s

    has_team = "Team" in raw_items
    has_org  = "Org"  in raw_items
    org_team_key = "Org" if has_org else ("Team" if has_team else None)

    ordered = []
    ordered.append(("Name", pretty_name))
    if "Pos" in raw_items: ordered.append(("Pos", raw_items["Pos"]))
    if org_team_key:       ordered.append((org_team_key, raw_items[org_team_key]))
    if "Current Level" in raw_items: ordered.append(("Current Level", raw_items["Current Level"]))

    bio_keys = ["Age","Ht","Wt","B","T","Sign Yr"]
    for bk in bio_keys:
        if bk in raw_items:
            ordered.append((bk, raw_items[bk]))

    skip_keys = {
        "Name","name_key","Name_key","PlayerId","PlayerId_best","__clean_allcols__","embedding","report","Report", "role"
        "Hit","Game Pwr","Raw Pwr","Spd","Fld","FB","CB","SL","CH","CMD","Hard Hit%","Sits","Tops","FB Type","Con Style", "Level", "Team", "Name_hit", "PID_unified_hit"
    }
    used_keys = {k for k, _ in ordered}
    if org_team_key: used_keys.add(org_team_key)
    used_keys.update(bio_keys)

    remaining = []
    for k, v in raw_items.items():
        if k in skip_keys or k in used_keys: continue
        remaining.append((k, v))
    remaining.sort(key=lambda t: t[0].lower())
    ordered.extend(remaining)

    # ── Nearest MLB comp (unchanged) ─────────────────────────────────────────
    comp = nearest_comp(emb4)
    # Build Scouting (Key Fields) with role-specific tools
    scout_base = ["Top 100","Name","Org","Pos","Current Level","Age"]
    if is_h:
        scout_tools = ["Hit","Game Pwr","Raw Pwr","Fld"]      # hitter tools
    else:
        scout_tools = ["FB","SL","CB","CH","CMD","Sits","Tops"]     # pitcher tools

    scout_cols = [c for c in (scout_base + scout_tools) if c in row]
    scout_row  = [str(row.get(c, "")) for c in scout_cols]
    return {
        "role": ("B" if is_h else "P"),
        "topline": {
            "mlb_prob_combined": round(mlb_comb * 100, 2),
            "mlb_yes_no": mlb_yes_no,
            "mlb_stats_prob": round(float(ps2_stats[1]) * 100, 2),
            "mlb_scout_prob": round(float(ps2_scout[1]) * 100, 2),
            "temps": {
                "T2_stats": round(T2_stats, 3),
                "T2_scout": round(T2_scout, 3),
                "T4_stats": round(T4_stats, 3),
                "T4_scout": round(T4_scout, 3),
            }
        },
        "four_class": {
            "ev_from_logits": int(ev_from_logits),
            "ev_display": ev_display,
            "predicted_class": int(cls_idx + 1),
            # DISPLAY bars use the tilted variants
            "probs_stats": [float(x) for x in p4_stats_vis.tolist()],
            "probs_scout": [float(x) for x in p4_scout_vis.tolist()],
            # Combined kept original (not shown as bars)
            "probs_comb": [float(x) for x in p4_comb.tolist()],
        },
        "mlb_comp": comp,
        "tables": {
            "stats_kv": ordered,
        },
        "scout": {
            "columns": scout_cols,
            "row": scout_row,
        },
        "raw": {
            "name": pretty_name,
            "report": report_text,
        }
    }

# ── Flask / Template (WHITE THEME) ───────────────────────────────────────────
app = Flask(__name__)

def dollar(x): return "${:,.0f}".format(x)
app.jinja_env.filters["dollar"] = dollar

HTML = """<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\"/>
    <title>Prospect Predictor — MLB Probability & Projected Earnings</title>
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
    <style>
      :root{
        --bg:#ffffff; --panel:#ffffff; --muted:#6b7280; --text:#0f172a;
        --accent:#2563eb; --accent2:#06b6d4; --border:#e5e7eb; --field:#f8fafc;
      }
      *{box-sizing:border-box}
      body{
        margin:0; font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;
        background:var(--bg); color:var(--text);
      }
      .wrap{ max-width:960px; margin:24px auto; padding:0 14px; }
      .title{ font-size:26px; font-weight:800; letter-spacing:.2px }
      .sub{ color:var(--muted); margin:6px 0 16px; font-size:14px }
      .panel{ background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:14px; }
      .row{ display:grid; grid-template-columns: 1fr; gap:14px }
      label{ display:block; font-size:12px; color:var(--muted); margin-bottom:6px }
      input[type=text], button{ font:inherit; color:inherit; }
      input[type=text]{
        width:100%; padding:10px 12px; border-radius:10px; border:1px solid var(--border);
        background:var(--field);
      }
      .btn{
        background:linear-gradient(90deg,var(--accent),var(--accent2)); color:#fff; border:none;
        padding:10px 14px; border-radius:10px; font-weight:800; cursor:pointer
      }
      .section{ margin-top:12px; padding:12px; border-radius:12px; background:#fcfcfd; border:1px solid var(--border); font-size:14px }
      .muted{ color:var(--muted); font-size:12px }
      .kpi{ display:flex; gap:12px; flex-wrap:wrap }
      .card{ flex:1 1 220px; background:#ffffff; border:1px solid var(--border); padding:12px; border-radius:12px }
      .label{ font-size:12px; color:#475569 }
      .value{ font-size:20px; font-weight:800; margin-top:4px }
      .grid2{ display:grid; grid-template-columns:1fr 1fr; gap:12px }
      @media (max-width: 700px){ .grid2{ grid-template-columns:1fr } }
      .bars{ display:grid; grid-template-columns:1fr; gap:8px; margin-top:8px }
      .bar{ background:#eef2ff; border:1px solid var(--border); border-radius:10px; overflow:hidden }
      .barfill{ height:10px; background:linear-gradient(90deg,var(--accent),var(--accent2)); width:0% }
      .barlabel{ display:flex; justify-content:space-between; font-size:12px; color:#334155; margin-bottom:4px }
      .kvgrid{ display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:8px; margin-top:8px }
      .kv{ background:#ffffff; border:1px solid var(--border); border-radius:10px; padding:8px }
      .kv .k{ font-size:11px; color:#64748b; margin-bottom:4px }
      .kv .v{ font-size:14px; font-weight:700; word-break:break-word }
      .selected{ margin-top:6px; font-size:14px; font-weight:700; color:#0f172a }
      .table-wrap{ margin-top:10px; border:1px solid var(--border); border-radius:12px; overflow:hidden; background:#ffffff }
      table{ border-collapse:collapse; width:100%; font-size:12px; table-layout:fixed }
      thead th{ position:sticky; top:0; z-index:2; background:#f8fafc; color:#334155; border-bottom:1px solid var(--border) }
      th, td{ padding:6px 8px; border-bottom:1px solid var(--border); text-align:left; white-space:normal; word-break:break-word }
      tr:hover td{ background:#f9fafb }
      pre.report{ white-space:pre-wrap; font-family:inherit; font-size:14px; line-height:1.5; color:#111827; margin:8px 0 0 }
      .combo-row{ display:grid; grid-template-columns: 1fr auto; gap:8px; align-items:end }
      .small{ font-size:12px; color:#6b7280 }
    </style>
    <script>
      function selectFromDatalist() {
        const input = document.getElementById('playerCombo');
        const hidden = document.getElementById('playerHidden');
        hidden.value = input.value;
      }
    </script>
  </head>
  <body>
    <div class=\"wrap\">
      <div class=\"title\">Prospect Predictor — MLB Probability & Projected Earnings</div>
      <div class=\"sub\">Search or pick a player below, then click Predict.</div>

      <div class=\"panel\">
        <form id=\"predictForm\" method=\"POST\" action=\"/\" onsubmit=\"selectFromDatalist()\">
          <div class=\"row\">

            <div class=\"combo-row\">
              <div>
                <label>Player ({{ names|length }} total)</label>
                <!-- Single combo input (search or dropdown) -->
                <input type=\"text\" id=\"playerCombo\" list=\"playerList\" placeholder=\"Start typing a name…\" autocomplete=\"off\" value=\"{{ cur_player or '' }}\" />
                <datalist id=\"playerList\">
                  {% for n in names %}
                    <option value=\"{{n}}\"></option>
                  {% endfor %}
                </datalist>
                <!-- Hidden field the server reads -->
                <input type=\"hidden\" id=\"playerHidden\" name=\"player\" />
                {% if cur_player %}
                  <div class=\"selected\">Selected: {{ cur_player }}</div>
                {% else %}
                  <div class=\"small\">Type to search or click the input’s dropdown caret to choose.</div>
                {% endif %}
              </div>
              <button class=\"btn\" type=\"submit\" name=\"action\" value=\"predict\">Predict</button>
            </div>

            {% if error %}
              <div class=\"section\" style=\"background:#fff7f7;border-color:#fecaca;color:#7f1d1d\">{{ error }}</div>
            {% endif %}

            {% if result %}
              <!-- MLB Probability -->
              <div class=\"section\">
                <div class=\"kpi\">
                  <div class=\"card\">
                    <div class=\"label\">MLB Probability 
                    </div>
                    <div class=\"value\">{{ result.topline.mlb_prob_combined }}% — <span>{{ result.topline.mlb_yes_no }}</span></div>
                  </div>
                </div>
              </div>

              <!-- 4-class Projection -->
              <div class="section">
                <div class="kpi">
                    <div class="card">
                    <div class="label">Projected Earnings (EV)</div>
                    <div class="value">{{ result.four_class.ev_display }}</div>
                    </div>
                </div>

               <style>
                .barrow { margin: 10px 0; }
                .barhdr { display:flex; justify-content:space-between; font-size:12px; color:#334155; margin-bottom:4px }
                .barstack { position:relative; width:100%; height:12px; border:1px solid var(--border); border-radius:10px; overflow:hidden; background:#eef2ff; }
                .seg { position:absolute; top:0; bottom:0; width:0 }
                .seg.stats { background:#3b82f6; }  /* blue */
                .seg.scout { background:#ef4444; }  /* red */
                </style>

                <div class="card">
                <div class="label">4-Bucket Confidence — Stats (blue) + Scout (red)</div>
                <div id="stacked-bars"></div>
                </div>

                <script>
                (function() {
                    // Inputs from backend (already display-tilted in your server)
                    const probsStats = {{ result.four_class.probs_stats | tojson }};
                    const probsScout = {{ result.four_class.probs_scout | tojson }};

                    // Prefer server-provided bucket labels; fallback if missing
                    const fallbackLabels = ["<$1,000,000", "$1,000,000–$5,000,000", "$5,000,000–$50,000,000", ">$50,000,000"];
                    const labels = ({{ (result.four_class.bucket_labels if result.four_class.get('bucket_labels') else []) | tojson }});
                    const names = (labels && labels.length === probsStats.length) ? labels : fallbackLabels;

                    // Weights: make stats dominate; small boost for L3/L4 if runner-up
                    const ALPHA = 0.85;           // stats weight
                    const BETA  = 0.15;           // scout weight
                    const SECOND_PLACE_BOOST = 1.20;

                    const k = probsStats.length;
                    let raw = new Array(k).fill(0);
                    for (let i = 0; i < k; i++) raw[i] = ALPHA * probsStats[i] + BETA * probsScout[i];

                    const order = Array.from(raw.keys()).sort((a,b) => raw[b] - raw[a]);
                    const secondIdx = order[1];
                    if (secondIdx === 2 || secondIdx === 3) raw[secondIdx] *= SECOND_PLACE_BOOST;

                    const sumRaw = raw.reduce((a,b)=>a+b, 0) || 1;
                    const comb = raw.map(x => x / sumRaw);

                    const container = document.getElementById('stacked-bars');
                    container.innerHTML = '';

                    for (let i = 0; i < k; i++) {
                    const totalPct = comb[i] * 100;

                    // Internal split (stats vs scout) for the stacked bar
                    const denom = (ALPHA*probsStats[i] + BETA*probsScout[i]) || 1e-12;
                    const statsShare = (ALPHA*probsStats[i]) / denom;
                    const scoutShare = (BETA*probsScout[i]) / denom;

                    const statsPctWidth = totalPct * statsShare;
                    const scoutPctWidth = totalPct * scoutShare;

                    const row = document.createElement('div');
                    row.className = 'barrow';

                    const hdr = document.createElement('div');
                    hdr.className = 'barhdr';
                    hdr.innerHTML = `<span>${names[i] || ('Label ' + (i+1))}</span><span>${totalPct.toFixed(2)}%</span>`;

                    const bar = document.createElement('div');
                    bar.className = 'barstack';

                    // Stats segment (blue), left→right
                    const segStats = document.createElement('div');
                    segStats.className = 'seg stats';
                    segStats.style.left = '0';
                    segStats.style.width = statsPctWidth + '%';

                    // Scout segment (red), stacked after stats
                    const segScout = document.createElement('div');
                    segScout.className = 'seg scout';
                    segScout.style.left = statsPctWidth + '%';
                    segScout.style.width = scoutPctWidth + '%';

                    bar.appendChild(segStats);
                    bar.appendChild(segScout);

                    row.appendChild(hdr);
                    row.appendChild(bar);
                    container.appendChild(row);
                    }
                })();
                </script>

              <!-- MLB Comp (NEW) -->
              {% if result.mlb_comp %}
              <div class=\"section\">
                <div class=\"kpi\">
                  <div class=\"card\">
                    <div class=\"label\">Closest MLB Comp (by report embedding)</div>
                    <div class=\"value\">{{ result.mlb_comp.name }}</div>
                    <div class=\"small\" style=\"margin-top:4px\">
                      Cosine Similarity: {{ (result.mlb_comp.similarity)|round(4) }}
                    </div>
                  </div>
                </div>
              </div>
              {% endif %}

              <!-- Ordered Stats (intuitive block) -->
              {% if result.tables.stats_kv and result.tables.stats_kv|length > 0 %}
              <div class=\"section\">
                <div><strong>Stats ({{ 'Hitter' if result.role=='B' else 'Pitcher' }})</strong></div>
                <div class=\"kvgrid\">
                  {% for k,v in result.tables.stats_kv %}
                    <div class=\"kv\"><div class=\"k\">{{k}}</div><div class=\"v\">{{v}}</div></div>
                  {% endfor %}
                </div>
              </div>
              {% endif %}

              <!-- Scouting subset -->
              {% if result.scout.columns and result.scout.columns|length > 0 %}
              <div class=\"section\">
                <div><strong>Scouting (Key Fields)</strong></div>
                <div class=\"table-wrap\">
                  <table>
                    <thead><tr>{% for c in result.scout.columns %}<th>{{c}}</th>{% endfor %}</tr></thead>
                    <tbody><tr>{% for v in result.scout.row %}<td>{{v}}</td>{% endfor %}</tr></tbody>
                  </table>
                </div>
              </div>
              {% endif %}

              {% if result.raw.report %}
              <div class=\"section\">
                <div><strong>Scouting Report</strong></div>
                <pre class=\"report\">{{ result.raw.report }}</pre>
              </div>
              {% endif %}

            {% else %}
              <div class=\"muted\">Pick a player and hit Predict.</div>
            {% endif %}
          </div>
        </form>
      </div>
    </div>
  </body>
</html>
"""

# ── Flask routes ─────────────────────────────────────────────────────────────
@app.route("/", methods=["GET","POST"])
def home():
    names = KEPT_NAMES
    cur_player = None
    result = None
    error = None

    if request.method == "POST":
        cur_player = (request.form.get("player") or "").strip()
        action = request.form.get("action")
        if action == "predict" and cur_player:
            try:
                if cur_player not in INDEX_BY_NAME:
                    error = "Selected player is not in the filtered cohort."
                else:
                    result = predict_for_name(cur_player)
            except Exception as e:
                error = f"Prediction failed: {e}"

    return render_template_string(
        HTML,
        names=names,
        cur_player=cur_player,
        result=result,
        error=error,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
