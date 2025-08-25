# -*- coding: utf-8 -*-
# paper_pipeline_mba_vs_rl.py
# Compare: Traditional Market Basket Analysis (MBA) vs RL (Contextual Bandit, Q-learning)
# Add: Confidence intervals (bootstrap), multiple OPE estimators (SNIPS/DR), Ablations, demo plots

import os, argparse, json, math, numpy as np, pandas as pd
from collections import defaultdict, Counter
from typing import Dict, Tuple, Callable, List

# sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDClassifier, Ridge

# matplotlib (single-figure, default colors per tool guidance)
import matplotlib.pyplot as plt

# -----------------------
# CLI
# -----------------------
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", type=str, default="./data")
ap.add_argument("--outdir", type=str, default="./paper_outputs")
ap.add_argument("--sample_train", type=int, default=20000)
ap.add_argument("--sample_test", type=int, default=10000)
ap.add_argument("--transitions", type=int, default=120000)   # Max transitions used for Q-learning
ap.add_argument("--boots", type=int, default=200)            # Bootstrap rounds (paper suggests ≥300)
ap.add_argument("--reward", type=str, default="margin", choices=["margin","revenue"])
ap.add_argument("--also_revenue", action="store_true")       # Additionally run OPE on revenue
ap.add_argument("--k", type=int, default=5)                  # KMeans number of clusters
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--fast", action="store_true")               # Fast mode (smaller samples, smaller B)
ap.add_argument("--models", nargs="+", default=[
    "MBA-Markov","MBA-Assoc","Bandit-Cluster","Q-Full","Q-NoSeq","Q-NoCluster"
])
ap.add_argument("--by_cluster", action="store_true")         # Output per-cluster breakdown
args = ap.parse_args()
np.random.seed(args.seed)

os.makedirs(args.outdir, exist_ok=True)

# -----------------------
# 0) Load
# -----------------------
def load_csvs(data_dir: str):
    products     = pd.read_csv(os.path.join(data_dir,"products.csv"))
    customers    = pd.read_csv(os.path.join(data_dir,"customers.csv"))
    transactions = pd.read_csv(os.path.join(data_dir,"transactions.csv"), parse_dates=["timestamp"])
    logs         = pd.read_csv(os.path.join(data_dir,"policy_logs.csv"))
    return products, customers, transactions, logs

products, customers, transactions, logs = load_csvs(args.data_dir)

console_banner("DATA LOADED: SHAPES & SAMPLE COLUMNS")
print("products.shape    :", products.shape)
print("customers.shape   :", customers.shape)
print("transactions.shape:", transactions.shape)
print("policy_logs.shape :", logs.shape)
print("\ntransactions columns:", list(transactions.columns)[:12], "...")
print("policy_logs columns :", list(logs.columns)[:12], "...")
print("\ntransactions timestamp range:",
      transactions["timestamp"].min(), "→", transactions["timestamp"].max())

# (Optional) load point-estimate summary if present in the same folder
point_estimates_path = os.path.join(args.data_dir, "Point_estimates__SNIPS_DR_vs_logging_.csv")
if os.path.isfile(point_estimates_path):
    point_estimates = pd.read_csv(point_estimates_path)
    # Purpose: show any precomputed point estimates the reader may want to compare.
    console_banner("OPTIONAL: LOADED PRECOMPUTED POINT ESTIMATES (HEAD)")
    print(point_estimates.head())

# -----------------------
# 1) Clustering (MiniBatchKMeans)
# -----------------------
# -----------------------
# OneHotEncoder compatibility helper (handles old/new scikit-learn)
# Purpose: avoid version issues so the demo runs on most environments.
# -----------------------
def make_ohe():
    try:
        # scikit-learn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# -----------------------
# 1) Clustering (MiniBatchKMeans)
# Purpose: derive `cluster_km` for personalization and segment-aware policies.
# Print cluster size distribution for interpretation in console.
# -----------------------
def ensure_kmeans(customers: pd.DataFrame, K: int) -> pd.DataFrame:
    if "cluster_km" in customers.columns:
        return customers
    numeric_features   = ["age"]
    categorical_feats  = ["income_band","family_status","region","loyalty_tier","gender"]
    binary_features    = ["has_baby","pet_owner"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", make_ohe(), categorical_feats),
        ("bin", "passthrough", binary_features),
    ], remainder="drop", sparse_threshold=0.0)
    X = pre.fit_transform(customers[numeric_features + categorical_feats + binary_features])
    km = MiniBatchKMeans(n_clusters=K, random_state=args.seed, batch_size=2048, n_init=3, max_iter=100)
    customers["cluster_km"] = km.fit_predict(X)
    return customers

customers = ensure_kmeans(customers, args.k)

def merge_cluster(df: pd.DataFrame) -> pd.DataFrame:
    return df.merge(customers[["customer_id","cluster_km"]], on="customer_id", how="left", validate="many_to_one")

transactions = merge_cluster(transactions)
logs = merge_cluster(logs)

# Purpose: show cluster distribution to explain personalization scope.
console_banner("CLUSTER DISTRIBUTION (customers.cluster_km)")
print(customers["cluster_km"].value_counts(dropna=False).sort_index())

# Add rewards (margin/revenue), align from transactions by (basket_id, t)
tx_key = transactions[["basket_id","sequence_index","margin","revenue"]].rename(
    columns={"sequence_index":"t","margin":"tx_margin","revenue":"tx_revenue"})
logs = logs.merge(tx_key, on=["basket_id","t"], how="left")

# Click uplift for rewards (you may modify per paper definition)
logs["r_margin"]  = np.where(logs["clicked"]==1, logs["tx_margin"],  0.15*logs["tx_margin"])
logs["r_revenue"] = np.where(logs["clicked"]==1, logs["tx_revenue"], 0.15*logs["tx_revenue"])

# Purpose: print a few joined rows so readers see how rewards/contexts appear.
console_banner("POLICY LOGS SAMPLE (joined rewards + contexts)")
print(logs[["basket_id","t","customer_id","cluster_km",
            "last_cat","dayofweek","promo_any","action_rec","action_rec_prob",
            "purchased_cat","clicked","r_margin","r_revenue"]].head(10).to_string(index=False))

# -----------------------
# 2) Sampling (train/test for OPE; transitions for Q)
# Purpose: show sizes for reproducibility and to explain experiment splits.
# -----------------------
def small_if_fast(v):
    if args.fast:
        return max(1000, int(v/3))
    return v

sample_train = small_if_fast(args.sample_train)
sample_test  = small_if_fast(args.sample_test)
max_trans    = small_if_fast(args.transitions)
boots        = small_if_fast(args.boots)

train_logs = logs.sample(min(sample_train, len(logs)), random_state=args.seed+1)
test_logs  = logs.sample(min(sample_test,  len(logs)), random_state=args.seed+2)

console_banner("SPLITS & HYPERPARAMS")
print(f"train_logs: {len(train_logs)} rows | test_logs: {len(test_logs)} rows")
print(f"max_transitions for Q: {max_trans} | bootstrap rounds: {boots}")

# Category universe
CATS = sorted(transactions["category"].astype(str).unique().tolist())
cat_to_id = {c:i for i,c in enumerate(CATS)}
global_majority = transactions["category"].astype(str).value_counts().idxmax()
print(f"\n#categories: {len(CATS)} | global_majority: {global_majority}")

# -----------------------
# 3) MBA baselines
# Purpose: build transparent references; print a few example transitions/rules for tutorial value.
# -----------------------
def build_markov(transactions: pd.DataFrame) -> Dict[str,Dict[str,float]]:
    counts = defaultdict(Counter)
    for _, g in transactions.sort_values(["basket_id","sequence_index"]).groupby("basket_id"):
        last = "START"
        for _, row in g.iterrows():
            nxt = str(row["category"])
            counts[last][nxt] += 1
            last = nxt
    probs = {}
    V = len(CATS)
    for last, ctr in counts.items():
        total = sum(ctr.values()) + V
        probs[last] = {a:(ctr.get(a,0)+1)/total for a in CATS}  # Laplace smoothing
    return probs

MARKOV = build_markov(transactions)

def actions_markov(df: pd.DataFrame) -> np.ndarray:
    out = []
    for x in df["last_cat"].astype(str).values:
        d = MARKOV.get(x)
        out.append(max(d.items(), key=lambda kv: kv[1])[0] if d else global_majority)
    return np.array(out, dtype=object)

def build_assoc(transactions: pd.DataFrame) -> Dict[str,str]:
    baskets = transactions.groupby("basket_id")["category"].apply(lambda s: list(map(str,s.values)))
    from collections import Counter as Ctr
    item_b = Ctr(); pair_b = defaultdict(int)
    for items in baskets:
        uniq = sorted(set(items))
        for a in uniq: item_b[a] += 1
        for i in range(len(uniq)):
            for j in range(len(uniq)):
                if i==j: continue
                pair_b[(uniq[i],uniq[j])] += 1
    nB = len(baskets); assoc_map = {}
    for a in CATS:
        best = None; best_score = -1e9
        for b in CATS:
            if a==b: continue
            c_ab = pair_b.get((a,b),0); c_a = item_b.get(a,0); c_b = item_b.get(b,0)
            if c_a==0 or c_b==0: continue
            p_ab=c_ab/nB; p_a=c_a/nB; p_b=c_b/nB
            conf=c_ab/c_a; lift=p_ab/(p_a*p_b+1e-9)
            score=conf*np.log(lift+1e-9)  # simple scoring
            if score>best_score:
                best_score=score; best=b
        assoc_map[a]= best if best is not None else global_majority
    assoc_map["START"]=global_majority
    return assoc_map

ASSOC = build_assoc(transactions)
def actions_assoc(df: pd.DataFrame) -> np.ndarray:
    return np.array([ASSOC.get(str(x), global_majority) for x in df["last_cat"].astype(str).values], dtype=object)

# Purpose: print a few human-readable Markov transitions and association rules.
console_banner("MBA SNAPSHOTS")
print("Markov P(next|START) (top 5):")
mk_start = pd.Series(MARKOV.get("START", {})).sort_values(ascending=False).head(5)
print(mk_start.to_string())
print("\nAssociation suggestions for 5 random antecedents:")
for a in np.random.choice(CATS, size=min(5, len(CATS)), replace=False):
    print(f"  If last_cat='{a}' → recommend '{ASSOC.get(a, global_majority)}'")

# -----------------------
# 4) Contextual Bandit (clustered)
# Purpose: learn fast, per-segment next-step policies; print fitted clusters and class space.
# -----------------------
def prepare_ctx(df: pd.DataFrame):
    enc = make_ohe()
    # ensure clean dtypes for one-hot
    X = enc.fit_transform(
        df[["last_cat","dayofweek","promo_any"]]
        .astype({"last_cat":str, "dayofweek":int, "promo_any":int})
    )
    y = df["purchased_cat"].astype(str).values
    return X, y, enc

def fit_cluster_bandits(train_logs: pd.DataFrame, min_rows=300):
    models, encs = {}, {}
    for cl, g in train_logs.groupby("cluster_km"):
        if len(g) < min_rows:
            continue
        X, y, enc = prepare_ctx(g)
        clf = SGDClassifier(loss="log_loss", max_iter=800, tol=1e-3,
                            random_state=args.seed+3).fit(X, y)
        models[cl] = clf
        encs[cl] = enc
    return models, encs

CB_MODELS, CB_ENCS = fit_cluster_bandits(train_logs, min_rows=200 if args.fast else 400)

console_banner("BANDIT FIT SUMMARY")
print(f"Fitted per-cluster bandit models: {sorted(CB_MODELS.keys())}")
# Show class labels learned in one sample cluster (if available)
if len(CB_MODELS) > 0:
    any_cl = sorted(CB_MODELS.keys())[0]
    print(f"Example cluster {any_cl} classes:", CB_MODELS[any_cl].classes_[:10], "...")

def actions_bandit_cluster(df: pd.DataFrame) -> np.ndarray:
    out = pd.Series(index=df.index, dtype=object)
    for cl, g in df.groupby("cluster_km"):
        if cl in CB_MODELS:
            enc = CB_ENCS[cl]; clf = CB_MODELS[cl]
            X = enc.transform(g[["last_cat","dayofweek","promo_any"]])
            proba = clf.predict_proba(X); classes = clf.classes_
            out.loc[g.index] = classes[np.argmax(proba, axis=1)]
        else:
            out.loc[g.index] = global_majority
    return out.values

# -----------------------
# 5) Q-learning (with ablations)
# Purpose: learn long-term value; print number of unique states encountered for intuition.
# -----------------------
def build_transitions(tx: pd.DataFrame, mode="full", max_rows=120000):
    tx = tx.sort_values(["customer_id","basket_id","sequence_index"])
    trans=[]; s2i={}
    def sid(s):
        s2i.setdefault(s, len(s2i)); return s2i[s]
    cnt=0
    for _, g in tx.groupby(["customer_id","basket_id"]):
        g=g.sort_values("sequence_index"); last="START"
        for i,(_,row) in enumerate(g.iterrows()):
            if mode=="full":
                s=(int(row["cluster_km"]), str(last), int(row["timestamp"].dayofweek))
                s2=(int(row["cluster_km"]), str(row["category"]), int(row["timestamp"].dayofweek)) if i<len(g)-1 else None
            elif mode=="no_seq":
                s=(int(row["cluster_km"]), "START", int(row["timestamp"].dayofweek)); s2=s if i<len(g)-1 else None
            elif mode=="no_cluster":
                s=("NONE", str(last), int(row["timestamp"].dayofweek))
                s2=("NONE", str(row["category"]), int(row["timestamp"].dayofweek)) if i<len(g)-1 else None
            else:
                s=(int(row["cluster_km"]), str(last), int(row["timestamp"].dayofweek)); s2=None
            a = CATS.index(str(row["category"])); r=float(row["margin"])
            trans.append((sid(s), a, r, sid(s2) if s2 is not None else None))
            last=str(row["category"]); cnt+=1
            if cnt>=max_rows: break
        if cnt>=max_rows: break
    return trans, s2i

def train_q(transitions, nS, nA, epochs=2, alpha=0.1, gamma=0.9):
    Q=np.zeros((nS,nA), dtype=np.float32)
    for _ in range(epochs):
        for s,a,r,s2 in transitions:
            target = r + (gamma*np.max(Q[s2]) if s2 is not None else 0.0)
            Q[s,a] += alpha*(target - Q[s,a])
    return Q

def make_q_policy(mode: str, max_rows: int) -> Callable[[pd.DataFrame], np.ndarray]:
    trans, s2i = build_transitions(transactions, mode=mode, max_rows=max_rows)
    Q = train_q(trans, len(s2i), len(CATS), epochs=2)
    def act(df: pd.DataFrame) -> np.ndarray:
        out=[]
        for _,r in df.iterrows():
            if mode=="no_seq":
                s=(int(r["cluster_km"]), "START", int(r["dayofweek"]))
            elif mode=="no_cluster":
                s=("NONE", str(r["last_cat"]), int(r["dayofweek"]))
            else:
                s=(int(r["cluster_km"]), str(r["last_cat"]), int(r["dayofweek"]))
            sid = s2i.get(s)
            out.append(CATS[int(np.argmax(Q[sid]))] if sid is not None else global_majority)
        return np.array(out, dtype=object)
    return act

Q_FULL       = make_q_policy("full",      max_rows=max_trans)
Q_NOSEQ      = make_q_policy("no_seq",    max_rows=max_trans)
Q_NOCLUSTER  = make_q_policy("no_cluster",max_rows=max_trans)

# Purpose: print #states for intuition (rebuilt briefly with smaller cap to avoid extra cost).
for mode in ["full","no_seq","no_cluster"]:
    _trans, _s2i = build_transitions(transactions, mode=mode, max_rows=min(max_trans, 30000))
    print(f"[Q-learning] mode={mode:10s} | #states≈{len(_s2i):5d} | sampled transitions={len(_trans):6d}")

# -----------------------
# 6) OPE: SNIPS / DR (+ confidence intervals)
# Purpose: define estimators; print chosen reward metric.
# -----------------------

# OneHotEncoder compatibility helper (handles old/new scikit-learn)
def make_ohe():
    try:
        # scikit-learn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def choose_reward_col():
    return "r_margin" if args.reward=="margin" else "r_revenue"

def snips(df: pd.DataFrame, actions: np.ndarray, reward_col: str, cap=10.0, idx=None) -> float:
    if idx is not None:
        df = df.iloc[idx]; actions = actions[idx]
    mask = (actions == df["action_rec"].astype(str).values).astype(float)
    p = df["action_rec_prob"].astype(float).values
    r = df[reward_col].astype(float).values
    w = 1.0/np.clip(p, 1e-3, 1.0)
    if cap is not None:
        w = np.minimum(w, cap)
    num = np.sum(mask*w*r); den = np.sum(mask*w)
    return float(num/den) if den > 0 else float("nan")

def fit_qhat(train_df: pd.DataFrame, reward_col: str):
    # ensure clean dtypes for one-hot encoders
    ctx = train_df[["last_cat","dayofweek","promo_any"]].astype(
        {"last_cat": str, "dayofweek": int, "promo_any": int}
    )
    act = train_df[["action_rec"]].astype(str)

    enc_ctx = make_ohe().fit(ctx)
    enc_act = make_ohe().fit(act)

    Xc = enc_ctx.transform(ctx)
    Xa = enc_act.transform(act)
    X  = np.hstack([Xc, Xa])

    y  = train_df[reward_col].astype(float).values
    model = Ridge(alpha=1.0).fit(X, y)
    return model, enc_ctx, enc_act

QH_MARGIN  = fit_qhat(train_logs, "r_margin")
QH_REVENUE = fit_qhat(train_logs, "r_revenue")

def dr(df: pd.DataFrame, actions: np.ndarray, reward_col: str, qh, cap=10.0, idx=None) -> float:
    if idx is not None:
        df = df.iloc[idx]; actions = actions[idx]
    model, enc_ctx, enc_act = qh
    ctx = df[["last_cat","dayofweek","promo_any"]].astype({"last_cat":str,"dayofweek":int,"promo_any":int})
    Xc  = enc_ctx.transform(ctx)
    Xa_l= enc_act.transform(df[["action_rec"]].astype(str)); q_log = model.predict(np.hstack([Xc,Xa_l]))
    Xa_p= enc_act.transform(pd.DataFrame({"action_rec": actions})); q_pi  = model.predict(np.hstack([Xc,Xa_p]))
    a_log = df["action_rec"].astype(str).values
    p     = df["action_rec_prob"].astype(float).values
    r     = df[reward_col].astype(float).values
    w     = 1.0/np.clip(p,1e-3,1.0)
    if cap is not None:
        w = np.minimum(w, cap)
    corr  = ((actions==a_log).astype(float)*w)*(r - q_log)
    return float(np.mean(q_pi + corr))

def cluster_bootstrap(df: pd.DataFrame, actions: np.ndarray, estimator="SNIPS",
                      reward_col="r_margin", B=200, seed=101) -> Tuple[float, Tuple[float,float]]:
    """
    Bootstrap over baskets using POSitional indices to avoid out-of-bounds with .iloc.
    Purpose: quantify uncertainty (95% CI) so we can discuss risk, not just point estimates.
    """
    rng = np.random.RandomState(seed)

    # Work in positional space
    df_pos = df.reset_index(drop=True)

    # Sanity: actions must align with df order/length
    actions = np.asarray(actions)
    if len(actions) != len(df_pos):
        raise ValueError(f"actions length {len(actions)} != df length {len(df_pos)}")

    # basket -> np.ndarray of POSITIONS
    basket_to_pos = df_pos.groupby("basket_id").indices
    baskets = np.array(list(basket_to_pos.keys()))

    vals = []
    for _ in range(B):
        samp = rng.choice(baskets, size=len(baskets), replace=True)
        idxs_list = [basket_to_pos[b] for b in samp if b in basket_to_pos]
        if not idxs_list:  # guard (empty)
            vals.append(np.nan)
            continue
        idxs = np.concatenate(idxs_list)

        if estimator == "DR":
            qh = QH_MARGIN if reward_col == "r_margin" else QH_REVENUE
            val = dr(df_pos, actions, reward_col, qh, cap=10.0, idx=idxs)
        else:
            val = snips(df_pos, actions, reward_col, cap=10.0, idx=idxs)
        vals.append(val)

    arr = np.array(vals, dtype=float)
    return float(np.nanmean(arr)), (float(np.nanpercentile(arr, 2.5)),
                                    float(np.nanpercentile(arr, 97.5)))
def baseline_ci(df: pd.DataFrame, reward_col: str, B=200, seed=201):
    """
    Baseline (logging policy) mean & CI, bootstrapped over baskets in positional space.
    Purpose: provide a direct, interpretable reference for OPE values.
    """
    rng = np.random.RandomState(seed)

    df_pos = df.reset_index(drop=True)
    basket_to_pos = df_pos.groupby("basket_id").indices
    baskets = np.array(list(basket_to_pos.keys()))

    vals = []
    for _ in range(B):
        samp = rng.choice(baskets, size=len(baskets), replace=True)
        idxs_list = [basket_to_pos[b] for b in samp if b in basket_to_pos]
        if not idxs_list:
            vals.append(np.nan)
            continue
        idxs = np.concatenate(idxs_list)
        vals.append(df_pos[reward_col].iloc[idxs].mean())

    arr = np.array(vals, dtype=float)
    return float(np.nanmean(arr)), (float(np.nanpercentile(arr, 2.5)),
                                    float(np.nanpercentile(arr, 97.5)))

# -----------------------
# 7) Evaluation (with Ablations and MBA)
# Purpose: compute SNIPS/DR (with CIs) per policy; print ranked table and quick summaries.
# -----------------------
def get_policy_actions(name: str, df: pd.DataFrame) -> np.ndarray:
    if name=="MBA-Markov":      return actions_markov(df)
    if name=="MBA-Assoc":       return actions_assoc(df)
    if name=="Bandit-Cluster":  return actions_bandit_cluster(df)
    if name=="Q-Full":          return Q_FULL(df)
    if name=="Q-NoSeq":         return Q_NOSEQ(df)
    if name=="Q-NoCluster":     return Q_NOCLUSTER(df)
    raise ValueError(f"Unknown model name: {name}")

def eval_all(models: List[str], df: pd.DataFrame, reward_col: str, boots: int):
    rows=[]
    # Precompute actions for each model
    acts_cache = {m: get_policy_actions(m, df) for m in models}
    for name in models:
        a = acts_cache[name]
        sn_m, sn_ci = cluster_bootstrap(df, a, estimator="SNIPS", reward_col=reward_col, B=boots, seed=301)
        dr_m, dr_ci = cluster_bootstrap(df, a, estimator="DR",    reward_col=reward_col, B=boots, seed=302)
        row = {
            "model": name,
            "SNIPS": round(sn_m,4), "SNIPS_CI_low": round(sn_ci[0],4), "SNIPS_CI_high": round(sn_ci[1],4),
            "DR":    round(dr_m,4), "DR_CI_low":    round(dr_ci[0],4), "DR_CI_high":    round(dr_ci[1],4),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("SNIPS", ascending=False).reset_index(drop=True)

reward_col = choose_reward_col()
console_banner(f"OPE EVALUATION START ({reward_col})")
results = eval_all(args.models, test_logs, reward_col, boots)

# baseline: direct reward of the logging policy
base_mean, base_ci = baseline_ci(test_logs, reward_col, B=boots, seed=401)
baseline_tbl = pd.DataFrame([{
    "model":"Logging (direct)",
    "reward_col": reward_col,
    "mean": round(base_mean,4),
    "CI_low": round(base_ci[0],4),
    "CI_high": round(base_ci[1],4),
}])

# Purpose: print the full results table and baseline reference for quick inspection.
console_banner("OPE RESULTS (SNIPS / DR with 95% CI)")
print(results.to_string(index=False))
print("\nLogging baseline (direct):", baseline_tbl.to_dict(orient="records")[0])

# -----------------------
# 8) Visualization (error bars)
# Purpose: render and save confidence-interval plots; also show them in Spyder's plot pane
# so you can immediately inspect figures while the files are saved for the paper.
# -----------------------
def plot_ci(df: pd.DataFrame, col: str, title: str, fn: str):
    xs = df["model"].tolist()
    vals = df[col].values
    lo = df[f"{col}_CI_low"].values
    hi = df[f"{col}_CI_high"].values
    yerr = np.vstack([vals - lo, hi - vals])

    plt.figure()
    plt.errorbar(xs, vals, yerr=yerr, fmt="o")
    plt.xticks(rotation=20, ha="right")
    plt.title(title)
    plt.ylabel(f"{col} ({reward_col})")
    plt.tight_layout()

    # Show in Spyder plot window (Purpose: immediate visual inspection during tutoring/demo)
    plt.show()

    # Save to disk for inclusion in the paper (Purpose: archival & reproducibility)
    plt.savefig(os.path.join(args.outdir, fn), dpi=160)
    plt.close()

plot_ci(results, "SNIPS", f"SNIPS with 95% CI ({reward_col})",
        "snips_margin_ci.png" if reward_col=="r_margin" else "snips_revenue_ci.png")
plot_ci(results, "DR",    f"DR with 95% CI ({reward_col})",
        "dr_margin_ci.png"   if reward_col=="r_margin" else "dr_revenue_ci.png")

# Ablation bar chart (Q-Full vs Q-NoSeq vs Q-NoCluster)
# Purpose: visualize contribution of sequence vs. clustering; shown in Spyder and saved to file.
def ablation_plot(df: pd.DataFrame):
    sub = df[df["model"].isin(["Q-Full","Q-NoSeq","Q-NoCluster"])].copy()
    if sub.empty: return
    xs = sub["model"].tolist()
    vals = sub["SNIPS"].values
    lo = sub["SNIPS_CI_low"].values
    hi = sub["SNIPS_CI_high"].values
    yerr = np.vstack([vals - lo, hi - vals])

    plt.figure()
    plt.bar(xs, vals)
    # Manual error bars
    for i,(x,v,l,h) in enumerate(zip(xs,vals,lo,hi)):
        plt.plot([i,i],[l,h])
    plt.title(f"Ablation (SNIPS, {reward_col})")
    plt.ylabel("Estimated reward")
    plt.tight_layout()

    # Show for on-the-spot explanation in Spyder
    plt.show()

    # Save for the paper
    plt.savefig(os.path.join(args.outdir,"ablation_bar.png"), dpi=160)
    plt.close()

ablation_plot(results)

# -----------------------
# 9) Save outputs
# Purpose: persist numeric summaries used in the paper/tables.
# -----------------------
results.to_csv(os.path.join(args.outdir,"ope_summary.csv"), index=False)
baseline_tbl.to_csv(os.path.join(args.outdir,"logging_baseline.csv"), index=False)

# Optional: also output a revenue-based evaluation (if specified)
if args.also_revenue and reward_col=="r_margin":
    results_rev = eval_all(args.models, test_logs, "r_revenue", boots)
    results_rev.to_csv(os.path.join(args.outdir,"ope_summary_revenue.csv"), index=False)
    plot_ci(results_rev, "SNIPS", "SNIPS with 95% CI (revenue)", "snips_revenue_ci.png")
    plot_ci(results_rev, "DR",    "DR with 95% CI (revenue)",    "dr_revenue_ci.png")

# -----------------------
# 10) Per-cluster breakdown (optional)
# Purpose: show segment-level behavior differences; saved to CSV for drill-down analysis.
# -----------------------
if args.by_cluster:
    rows=[]
    acts_cache = {m: get_policy_actions(m, test_logs) for m in args.models}
    for cl, g in test_logs.groupby("cluster_km"):
        for name in args.models:
            a = acts_cache[name][g.index.values]
            sn_m, sn_ci = cluster_bootstrap(g, a, estimator="SNIPS", reward_col=reward_col, B=max(boots//2, 60), seed=501+int(cl))
            rows.append({"cluster_km": int(cl), "model": name,
                         "SNIPS": round(sn_m,4), "CI_low": round(sn_ci[0],4), "CI_high": round(sn_ci[1],4), "n": len(g)})
    bycl = pd.DataFrame(rows)
    bycl.to_csv(os.path.join(args.outdir,"by_cluster_snips.csv"), index=False)
    # Purpose: quick console peek to understand segment differences.
    console_banner("BY-CLUSTER SNIPS SUMMARY (HEAD)")
    print(bycl.head().to_string(index=False))

print("\n=== DONE ===")
print("Outputs written to:", args.outdir)
print("Top by SNIPS:")
print(results[["model","SNIPS","SNIPS_CI_low","SNIPS_CI_high"]].head().to_string(index=False))
print("\nLogging (direct):", baseline_tbl.to_dict(orient="records")[0])
