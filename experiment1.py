from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import draft_main
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC, LinearSVC

SEED = 42
VAL_SIZE = 0.15

ALGORITHMS = ["SVM", "LR"]

QUICK_MODE = False
DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"
LOG_DIR = OUT_DIR / "logs"
for _d in (FIG_DIR, RES_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LOG = logging.getLogger("exp1")

def setup_logging(tag: str) -> Path:
    log_path = LOG_DIR / f"run_{tag}.log"
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    return log_path

def set_plot_style() -> None:
    import matplotlib.font_manager as fm
    for font_path in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if Path(font_path).exists():
            try:
                fm.fontManager.addfont(font_path)
            except Exception:
                pass
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.bbox"] = "tight"

def load_data():
    old_cwd = Path.cwd()
    try:
        os.chdir(DATA_DIR)
        X_train, y_train, X_test_unlabeled = draft_main.load_data()
    finally:
        os.chdir(old_cwd)
    train_df = pd.DataFrame({"text": X_train, "target": y_train})
    test_df = pd.DataFrame({"text": X_test_unlabeled})
    assert train_df["text"].notna().all() and test_df["text"].notna().all()
    return train_df, test_df

def run_eda(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    y = train_df["target"]
    word_len = train_df["text"].str.split().apply(len)
    dup = int(train_df["text"].duplicated().sum())
    stats = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_classes": int(y.nunique()),
        "class_counts": {int(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "class_ratio_max_min": round(
            float(y.value_counts().max() / y.value_counts().min()), 4
        ),
        "char_len_mean": round(float(train_df["text"].str.len().mean()), 1),
        "char_len_median": round(float(train_df["text"].str.len().median()), 1),
        "word_len_mean": round(float(word_len.mean()), 1),
        "word_len_max": int(word_len.max()),
        "duplicated_texts": dup,
        "test_len_mean": round(float(test_df["text"].str.len().mean()), 1),
    }
    LOG.info("数据探索: %s", json.dumps(stats, ensure_ascii=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    counts = y.value_counts().sort_index()
    axes[0].bar(counts.index.astype(str), counts.values, color="#4C72B0")
    axes[0].set_title("训练集类别分布（10 类，基本均衡）")
    axes[0].set_xlabel("类别标签")
    axes[0].set_ylabel("样本数")
    axes[0].axhline(counts.mean(), color="#C44E52", ls="--", lw=1, label="均值")
    axes[0].legend()
    axes[1].hist(np.clip(word_len, 0, 800), bins=50, color="#55A868")
    axes[1].set_title("训练文本词数分布（截断至 800 词）")
    axes[1].set_xlabel("词数")
    axes[1].set_ylabel("文档数")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_data_overview.png")
    plt.close(fig)

    (RES_DIR / "eda_summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats

_HEADER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]{1,20}:")
_QUOTE_RE = re.compile(r"^\s*>.*$", re.M)
_SIG_RE = re.compile(r"^\s*--\s*$")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_NUM_RE = re.compile(r"\b\d[\d.,]*\b")

def strip_headers(text: str) -> str:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == "":
            head = "\n".join(lines[:i])
            if _HEADER_RE.search(head):
                return "\n".join(lines[i + 1:])
            return text
    return text

def strip_quotes(text: str) -> str:
    text = _QUOTE_RE.sub("", text)
    lines = text.split("\n")
    out = []
    for line in lines:
        if _SIG_RE.match(line):
            break
        out.append(line)
    return "\n".join(out)

def normalize_tokens(text: str) -> str:
    text = _EMAIL_RE.sub(" emailtoken ", text)
    text = _URL_RE.sub(" urltoken ", text)
    text = _NUM_RE.sub(" numtoken ", text)
    return text

PREPROCESSORS = {
    "raw": lambda t: t,
    "noquote_only": strip_quotes,
    "noheader": strip_headers,
    "noheader_noquote": lambda t: strip_quotes(strip_headers(t)),
    "noheader_norm": lambda t: normalize_tokens(strip_headers(t)),
    "noheader_noquote_norm": lambda t: normalize_tokens(strip_quotes(strip_headers(t))),
}

def make_preprocessor(name: str):
    fn = PREPROCESSORS[name]
    return lambda texts: [fn(t) for t in texts]

VECTORIZERS = {

    "V0_draft_5k": dict(max_features=5000),

    "V1_word20k": dict(max_features=20000, min_df=2, sublinear_tf=True),

    "V2_word20k_12gram": dict(
        max_features=50000, min_df=2, sublinear_tf=True, ngram_range=(1, 2)
    ),

    "V3_word20k_stop": dict(
        max_features=20000, min_df=2, sublinear_tf=True, stop_words="english"
    ),

    "V4_word5k_sublinear": dict(max_features=5000, min_df=2, sublinear_tf=True),
}

def make_vectorizer(name: str):
    if name == "V5_wordchar_union":
        return make_union_vectorizer()
    return TfidfVectorizer(**VECTORIZERS[name])

class VectorCache:

    def __init__(self, prep: str, vec: str):
        self.prep = prep
        self.vec = vec
        self.vectorizer = None

    def fit(self, X_tr, X_va):
        t0 = time.time()
        pre = make_preprocessor(self.prep)
        self.vectorizer = make_vectorizer(self.vec)
        A_tr = self.vectorizer.fit_transform(pre(X_tr))
        A_va = self.vectorizer.transform(pre(X_va))
        self.fit_time = time.time() - t0
        self.n_features = A_tr.shape[1]
        self.matrices = (A_tr, A_va)
        return self.matrices

def build_model(family: str, params: dict):
    params = dict(params)
    if isinstance(params.get("hidden_layer_sizes"), list):
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    if family == "NB":
        return MultinomialNB(**params)
    if family == "SVM":
        return SVC(**{**params, "random_state": SEED})
    if family == "LinearSVM":
        return LinearSVC(**{**params, "random_state": SEED})
    if family == "LR":
        return LogisticRegression(**{**params, "random_state": SEED})
    if family == "MLP":
        return MLPClassifier(**{**params, "random_state": SEED})
    raise ValueError(f"未知算法族: {family}")

def evaluate_model(model, A_tr, y_tr, A_va, y_va) -> dict:
    t0 = time.time()
    model.fit(A_tr, y_tr)
    fit_time = time.time() - t0
    pred = model.predict(A_va)
    return {
        "val_accuracy": round(float(accuracy_score(y_va, pred)), 4),
        "val_macro_f1": round(float(f1_score(y_va, pred, average="macro")), 4),
        "val_weighted_f1": round(float(f1_score(y_va, pred, average="weighted")), 4),
        "fit_seconds": round(fit_time, 2),
    }

def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj

def cfg_key(row: dict) -> str:
    params = row["params"]
    if isinstance(params, str):
        params = json.loads(params)
    parts = [row["stage"], row["prep"], row["vectorizer"], row["family"]]
    parts.append(json.dumps(_jsonable(params), sort_keys=True, ensure_ascii=False))
    return "|".join(parts)

def load_results(path: Path) -> list:
    if path.exists():
        return pd.read_csv(path).to_dict("records")
    return []

def save_results(rows: list, path: Path) -> None:
    out = []
    for r in rows:
        r = dict(r)
        if "params" in r and not isinstance(r["params"], str):
            r["params"] = json.dumps(_jsonable(r["params"]), ensure_ascii=False)
        out.append(r)
    pd.DataFrame(out).to_csv(path, index=False, encoding="utf-8-sig")

def run_configs(configs, path: Path, evaluate_fn, cache: dict):
    rows = load_results(path)
    done = {cfg_key(r) for r in rows}
    for i, cfg in enumerate(configs, 1):
        key = cfg_key(cfg)
        if key in done:
            LOG.info("[%d/%d] 跳过已完成配置: %s", i, len(configs), key)
            continue
        row = dict(cfg)
        code = (cfg["prep"], cfg["vectorizer"])
        if code not in cache:
            vc = VectorCache(*code)
            vc.fit(evaluate_fn["X_tr"], evaluate_fn["X_va"])
            cache[code] = vc
        vc = cache[code]
        row["n_features"] = vc.n_features
        row["vectorize_seconds"] = round(vc.fit_time, 2)
        A_tr, A_va = vc.matrices
        model = build_model(cfg["family"], cfg["params"])
        row.update(evaluate_model(model, A_tr, evaluate_fn["y_tr"], A_va, evaluate_fn["y_val"]))
        LOG.info(
            "[%d/%d] %s(%s) acc=%.4f macroF1=%.4f (%.1fs)",
            i, len(configs), cfg["family"], json.dumps(cfg["params"], ensure_ascii=False),
            row["val_accuracy"], row["val_macro_f1"], row["fit_seconds"],
        )
        rows.append(row)
        save_results(rows, path)
        done.add(key)
    return rows

def stage_representation(ctx, quick: bool):
    LOG.info("=" * 30 + " 阶段一：预处理 / 特征表示对照实验 " + "=" * 30)
    probe = {"family": "LinearSVM", "params": {"C": 1.0}}
    prep_list = ["raw", "noquote_only", "noheader", "noheader_noquote",
                 "noheader_norm", "noheader_noquote_norm"]
    vec_list = ["V0_draft_5k", "V1_word20k", "V2_word20k_12gram", "V3_word20k_stop",
                "V4_word5k_sublinear"]
    if quick:
        prep_list = ["raw", "noheader_noquote"]
        vec_list = ["V0_draft_5k", "V1_word20k"]

    path = RES_DIR / "stage1_representation.csv"

    prep_configs = [
        dict(stage="repr_prep", prep=p, vectorizer="V1_word20k", **probe) for p in prep_list
    ]
    run_configs(prep_configs, path, ctx["eval"], ctx["cache"])
    df = pd.read_csv(path)
    sub = df[(df.stage == "repr_prep")].dropna(subset=["val_macro_f1"])
    best_prep = sub.sort_values(["val_macro_f1", "val_accuracy"], ascending=False).iloc[0]
    LOG.info("预处理对照实验最佳: %s (macroF1=%.4f)",
             best_prep["prep"], best_prep["val_macro_f1"])

    vec_configs = [
        dict(stage="repr_vec", prep=best_prep["prep"], vectorizer=v, **probe)
        for v in vec_list
    ]
    run_configs(vec_configs, path, ctx["eval"], ctx["cache"])
    df = pd.read_csv(path)
    sub = df[(df.stage == "repr_vec")].dropna(subset=["val_macro_f1"])
    best_vec = sub.sort_values(["val_macro_f1", "val_accuracy"], ascending=False).iloc[0]
    LOG.info("最佳预处理: %s (macroF1=%.4f)", best_prep["prep"], best_prep["val_macro_f1"])
    LOG.info("最佳表示: %s (macroF1=%.4f)", best_vec["vectorizer"], best_vec["val_macro_f1"])

    combo_prep, combo_vec = best_prep["prep"], best_vec["vectorizer"]
    combo = dict(stage="repr_combo", prep=combo_prep, vectorizer=combo_vec, **probe)
    rows = run_configs([combo], RES_DIR / "stage1_representation.csv", ctx["eval"], ctx["cache"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    sub = df[df.stage == "repr_prep"].dropna(subset=["val_macro_f1"]).sort_values("val_macro_f1")
    axes[0].barh(sub["prep"], sub["val_macro_f1"], color="#4C72B0")
    axes[0].set_xlim(sub["val_macro_f1"].min() - 0.02, 1.0)
    axes[0].set_title("预处理对照实验（LinearSVM + TF-IDF 20k）")
    axes[0].set_xlabel("验证集宏 F1")
    sub = df[df.stage == "repr_vec"].dropna(subset=["val_macro_f1"]).sort_values("val_macro_f1")
    axes[1].barh(sub["vectorizer"], sub["val_macro_f1"], color="#55A868")
    axes[1].set_xlim(sub["val_macro_f1"].min() - 0.02, 1.0)
    axes[1].set_title("特征表示对照实验（词表 / n-gram / 停用词）")
    axes[1].set_xlabel("验证集宏 F1")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_representation_ablation.png")
    plt.close(fig)
    return combo_prep, combo_vec

TUNING_GRIDS = {

    "SVM": [
        {"kernel": "linear", "C": c} for c in (0.1, 1.0, 10.0, 100.0)
    ] + [
        {"kernel": "rbf", "C": c, "gamma": "scale"} for c in (1.0, 10.0)
    ],

    "LinearSVM": [
        {"C": c, "max_iter": 5000} for c in (0.1, 1.0, 10.0, 100.0)
    ],

    "LR": [
        {"C": c, "max_iter": 1000, "solver": "lbfgs"} for c in (0.1, 1.0, 10.0, 100.0)
    ],
}

def normalize_params(params: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}

def stage_tuning(ctx, prep: str, vec: str, quick: bool):
    LOG.info("=" * 30 + " 阶段二：超参数调优 " + "=" * 30)
    grids = {k: list(v) for k, v in TUNING_GRIDS.items()}
    if quick:
        grids = {
            "SVM": [{"kernel": "linear", "C": 1.0}],
            "LinearSVM": [{"C": 1.0, "max_iter": 5000}],
            "LR": [{"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}],
        }

    configs = []
    for family, plist in grids.items():
        for params in plist:
            p = dict(params)
            if "hidden_layer_sizes" in p:
                p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
            configs.append(dict(stage="tuning", prep=prep, vectorizer=vec,
                                family=family, params=normalize_params(p)))
    rows = run_configs(configs, RES_DIR / "stage2_tuning.csv", ctx["eval"], ctx["cache"])
    ok = [r for r in rows if r.get("val_macro_f1") == r.get("val_macro_f1")]
    best = max(ok, key=lambda r: (r["val_macro_f1"], r["val_accuracy"]))
    LOG.info("调优阶段最佳: %s %s (acc=%.4f macroF1=%.4f)", best["family"],
             best["params"], best["val_accuracy"], best["val_macro_f1"])
    return best

def make_union_vectorizer():
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(max_features=30000, min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     min_df=3, sublinear_tf=True, max_features=30000)),
        ]
    )

def build_estimator(family: str, params: dict):
    if family == "Stacking":
        from sklearn.ensemble import StackingClassifier

        base = [
            ("svm", LinearSVC(C=params.get("svm_C", 1.0), random_state=SEED)),
            ("lr", LogisticRegression(C=params.get("lr_C", 10.0), max_iter=1000,
                                      random_state=SEED)),
        ]
        return StackingClassifier(
            estimators=base,
            final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
            cv=params.get("cv", 5),
            n_jobs=1,
            passthrough=False,
        )
    return build_model(family, params)

def stage_innovation(ctx, prep: str, best_cfg: dict, quick: bool):
    LOG.info("=" * 30 + " 阶段三：创新尝试 " + "=" * 30)
    path = RES_DIR / "stage3_innovation.csv"
    rows = load_results(path)
    done = {cfg_key(r) for r in rows}

    def probe(vec_name: str, family: str, params: dict, note: str):
        cfg = dict(stage="innovation", prep=prep, vectorizer=vec_name,
                   family=family, params=normalize_params(params))
        key = cfg_key(cfg)
        if key in done:
            LOG.info("跳过已完成: %s", key)
            return
        code = (prep, vec_name)
        if code not in ctx["cache"]:
            vc = VectorCache(*code)
            vc.matrices = vc.fit(ctx["eval"]["X_tr"], ctx["eval"]["X_va"])
            ctx["cache"][code] = vc
        vc = ctx["cache"][code]
        A_tr, A_va = vc.matrices
        row = dict(cfg)
        row["n_features"] = vc.n_features
        row["vectorize_seconds"] = round(vc.fit_time, 2)
        row["note"] = note
        model = build_estimator(family, params)
        row.update(evaluate_model(model, A_tr, ctx["eval"]["y_tr"], A_va,
                                  ctx["eval"]["y_val"]))
        LOG.info("[创新] %s + %s -> acc=%.4f macroF1=%.4f (%.1fs)", family, vec_name,
                 row["val_accuracy"], row["val_macro_f1"], row["fit_seconds"])
        rows.append(row)
        save_results(rows, path)
        done.add(key)

    probe("V1_word20k", "LinearSVM", {"C": 1.0}, "对照：纯词级 TF-IDF")
    probe("V5_wordchar_union", "LinearSVM", {"C": 1.0},
          "创新：词级 + 字符级 TF-IDF 联合表示")
    probe("V1_word20k", "Stacking", {"cv": 5, "svm_C": 1.0, "lr_C": 10.0},
          "创新：线性 SVM + 逻辑回归栈式集成（元学习器 = 逻辑回归）")
    if quick:
        return pd.read_csv(path)
    probe("V5_wordchar_union", "Stacking", {"cv": 5, "svm_C": 1.0, "lr_C": 10.0},
          "创新：联合表示 + 栈式集成")
    return pd.read_csv(path)

def stage_robustness(ctx, candidates: list, quick: bool):
    LOG.info("=" * 30 + " 阶段四：稳健性检验（5 折 × 3 种子） " + "=" * 30)
    path = RES_DIR / "stage4_robustness.csv"
    rows = load_results(path)
    seeds = [SEED] if quick else [SEED, SEED + 1, SEED + 2]
    X_all = np.asarray(ctx["X_all"], dtype=object)
    y_all = np.asarray(ctx["y_all"])
    for cfg in candidates:
        family = cfg["family"]
        if family in {"Stacking", "Vote"}:
            continue
        params = cfg["params"] if isinstance(cfg["params"], dict) else json.loads(cfg["params"])
        p = dict(params)
        if "hidden_layer_sizes" in p:
            p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        for seed in seeds:
            key = f"{family}|{json.dumps(normalize_params(p), sort_keys=True)}|{seed}"
            if any(f"{r['family']}|{r['params']}|{int(r['seed'])}" == key for r in rows):
                continue
            skf = StratifiedKFold(n_splits=5 if not quick else 3, shuffle=True,
                                  random_state=seed)
            scores, accs = [], []
            for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all), 1):
                vc = VectorCache(cfg["prep"], cfg["vectorizer"])
                A_tr, A_va = vc.fit(X_all[tr_idx].tolist(), X_all[va_idx].tolist())
                model = build_estimator(family, p)
                model.fit(A_tr, y_all[tr_idx])
                pred = model.predict(A_va)
                scores.append(f1_score(y_all[va_idx], pred, average="macro"))
                accs.append(accuracy_score(y_all[va_idx], pred))
            rows.append(dict(
                stage="robustness", prep=cfg["prep"], vectorizer=cfg["vectorizer"],
                family=family, params=json.dumps(normalize_params(p), sort_keys=True,
                                                 ensure_ascii=False),
                seed=seed, n_folds=int(skf.n_splits),
                macro_f1_mean=round(float(np.mean(scores)), 4),
                macro_f1_std=round(float(np.std(scores)), 4),
                accuracy_mean=round(float(np.mean(accs)), 4),
                accuracy_std=round(float(np.std(accs)), 4),
            ))
            LOG.info("[稳健性] %s seed=%d macroF1=%.4f±%.4f", family, seed,
                     np.mean(scores), np.std(scores))
            save_results(rows, path)
    return pd.read_csv(path)

def stage_error_analysis(ctx, best, prep, vec):
    LOG.info("=" * 30 + " 阶段五：错误分析 " + "=" * 30)
    cache_key = (prep, vec)
    if cache_key not in ctx["cache"]:
        vc = VectorCache(*cache_key)
        vc.matrices = vc.fit(ctx["eval"]["X_tr"], ctx["eval"]["X_va"])
        ctx["cache"][cache_key] = vc
    vc = ctx["cache"][cache_key]
    A_tr, A_va = vc.matrices
    y_tr, y_va = ctx["eval"]["y_tr"], ctx["eval"]["y_val"]
    params = best["params"] if isinstance(best["params"], dict) else json.loads(best["params"])
    p = dict(params)
    if "hidden_layer_sizes" in p:
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
    model = build_estimator(best["family"], p)
    model.fit(A_tr, y_tr)
    pred = model.predict(A_va)

    labels = sorted(np.unique(y_tr).tolist())
    cm = confusion_matrix(y_va, pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    report = classification_report(y_va, pred, digits=4)
    (RES_DIR / "best_classification_report.txt").write_text(
        f"最佳模型: {best['family']} {normalize_params(p)}\n"
        f"预处理: {prep}  特征: {vec}\n\n{report}", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, mat, title, fnt in (
        (axes[0], cm, "混淆矩阵（计数）", "%.0f"),
        (axes[1], cm_norm, "混淆矩阵（按真实类别归一化）", "%.2f"),
    ):
        im = ax.imshow(mat, cmap="Blues")
        pretty = ", ".join(f"{k}={v}" for k, v in normalize_params(p).items())
        ax.set_title(f"{title}\n{best['family']} ({pretty})", fontsize=10)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        thr = mat.max() / 2
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j] > 0:
                    ax.text(j, i, fnt % mat[i, j], ha="center", va="center",
                            fontsize=6.5, color="white" if mat[i, j] > thr else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_confusion_matrix.png")
    plt.close(fig)

    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if off[i, j] > 0:
                pairs.append((off[i, j], labels[i], labels[j]))
    pairs.sort(reverse=True)
    with (RES_DIR / "top_confusions.txt").open("w", encoding="utf-8") as f:
        f.write("真实类别 -> 预测类别 : 样本数\n")
        for n, a, b in pairs[:15]:
            f.write(f"  {a} -> {b} : {n}\n")

    X_va_raw = ctx["eval"]["X_va"]
    lengths = np.array([len(t.split()) for t in X_va_raw])
    buckets = [(0, 100), (100, 200), (200, 400), (400, 800), (800, 10 ** 9)]
    bnames, bacc, bn = [], [], []
    for lo, hi in buckets:
        m = (lengths >= lo) & (lengths < hi)
        if m.sum() > 0:
            bnames.append(f"{lo}-{hi if hi < 10**9 else '+'}")
            bacc.append(accuracy_score(np.asarray(y_va)[m], pred[m]))
            bn.append(int(m.sum()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].bar(bnames, bacc, color="#C44E52")
    for i, (v, n) in enumerate(zip(bacc, bn)):
        axes[0].text(i, v + 0.01, f"n={n}", ha="center", fontsize=8)
    axes[0].set_ylim(0.5, 1.02)
    axes[0].set_title("不同文本长度区间的验证准确率")
    axes[0].set_xlabel("词数区间")
    axes[0].set_ylabel("准确率")

    per_class = f1_score(y_va, pred, average=None, labels=labels)
    axes[1].bar([str(l) for l in labels], per_class, color="#8172B3")
    axes[1].set_ylim(0.6, 1.02)
    axes[1].set_title("各类别 F1（验证集）")
    axes[1].set_xlabel("类别")
    axes[1].set_ylabel("F1")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_error_analysis.png")
    plt.close(fig)

    err_idx = np.where(pred != np.asarray(y_va))[0]
    err_rows = []
    for i in err_idx[:200]:
        err_rows.append({
            "true": int(y_va[i]), "pred": int(pred[i]), "n_words": int(lengths[i]),
            "text_snippet": " ".join(str(X_va_raw[i]).split())[:300],
        })
    pd.DataFrame(err_rows).to_csv(RES_DIR / "misclassified_examples.csv",
                                  index=False, encoding="utf-8-sig")
    LOG.info("验证集错误分类样本 %d / %d，已保存错例清单与混淆矩阵",
             len(err_idx), len(y_va))
    return {"n_errors": int(len(err_idx)),
            "top_confusion": [(int(n), int(a), int(b)) for n, a, b in pairs[:5]]}

def stage_final(ctx, best, prep, vec, refit_full: bool = True):
    LOG.info("=" * 30 + " 阶段六：最终训练与预测提交 " + "=" * 30)
    X_all, y_all = ctx["X_all"], ctx["y_all"]
    X_test = ctx["X_test"]
    vc = VectorCache(prep, vec)
    A_all, A_test = vc.fit(X_all, X_test)
    params = best["params"] if isinstance(best["params"], dict) else json.loads(best["params"])
    p = dict(params)
    if "hidden_layer_sizes" in p:
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
    model = build_estimator(best["family"], p)
    t0 = time.time()
    model.fit(A_all, y_all)
    fit_seconds = time.time() - t0
    pred = model.predict(A_test)

    sub = pd.DataFrame({"target": pred.astype(int)})
    if not QUICK_MODE:
        sub_path = DATA_DIR / "predictions.csv"
        sub.to_csv(sub_path, index=False, header=False)

    dist = sub["target"].value_counts().sort_index()
    train_dist = pd.Series(y_all).value_counts(normalize=True).sort_index()
    test_dist = (dist / len(sub)).sort_index()
    cmp = pd.DataFrame({
        "train_ratio": train_dist.round(4),
        "test_pred_ratio": test_dist.round(4),
        "test_count": dist,
    })
    cmp.index.name = "class"
    cmp.to_csv(RES_DIR / "final_prediction_distribution.csv", encoding="utf-8-sig")

    meta = {
        "model": best["family"],
        "params": normalize_params(p),
        "preprocessing": prep,
        "vectorizer": vec,
        "n_features": int(vc.n_features),
        "refit_on": "train + validation (全部有标签数据)" if refit_full else "train",
        "n_train_final": int(len(X_all)),
        "n_test": int(len(X_test)),
        "val_accuracy": float(best["val_accuracy"]),
        "val_macro_f1": float(best["val_macro_f1"]),
        "fit_seconds": round(fit_seconds, 2),
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "python_version": platform.python_version(),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_file_rows": int(len(sub)),
        "prediction_class_range": [int(sub["target"].min()), int(sub["target"].max())],
    }
    (RES_DIR / "final_model_card.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOG.info("最终模型 %s %s | 特征 %d 维 | 预测文件 %d 行 | 用时 %.1fs",
             best["family"], normalize_params(p), vc.n_features, len(sub), fit_seconds)
    LOG.info("预测类别分布:\n%s", cmp.to_string())
    return meta

def parse_params(v):
    return v if isinstance(v, dict) else json.loads(v)

def _mpl(params):
    p = dict(params)
    for k in list(p):
        if isinstance(p[k], list):
            p[k] = tuple(p[k])
    return p

def fig_tuning_curves(quick: bool = False):
    df = pd.read_csv(RES_DIR / "stage2_tuning.csv")
    df["p"] = df["params"].apply(parse_params)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))

    ax = axes[0]
    for kernel, color, marker, ls in (("linear", "#C44E52", "o", "-"),
                                      ("rbf", "#8172B3", "s", "--")):
        sub = df[(df.family == "SVM")
                 & (df["p"].apply(lambda d: d.get("kernel") == kernel))].copy()
        if not len(sub):
            continue
        sub["x"] = sub["p"].apply(lambda d: d["C"])
        sub = sub.sort_values("x")
        ax.plot(sub["x"], sub["val_macro_f1"], marker=marker, ls=ls, color=color,
                label=f"{kernel} 核")
        for x, y in zip(sub["x"], sub["val_macro_f1"]):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6),
                        fontsize=6.5, ha="center")
    sub = df[df.family == "LinearSVM"].copy()
    if len(sub):
        sub["x"] = sub["p"].apply(lambda d: d["C"])
        sub = sub.sort_values("x")
        ax.plot(sub["x"], sub["val_macro_f1"], marker="D", ls=":",
                color="#4C72B0", label="liblinear")
        for x, y in zip(sub["x"], sub["val_macro_f1"]):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -12),
                        fontsize=6.5, ha="center", color="#4C72B0")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    ax.set_xlabel("C（惩罚系数，对数刻度）")
    ax.set_ylabel("验证集宏 F1")
    ax.set_title("支持向量机：实现、核函数与 C")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    ax = axes[1]
    sub = df[df.family == "LR"].copy()
    if len(sub):
        sub["x"] = sub["p"].apply(lambda d: d["C"])
        sub = sub.sort_values("x")
        ax.plot(sub["x"], sub["val_macro_f1"], "o-", color="#55A868")
        for x, y in zip(sub["x"], sub["val_macro_f1"]):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6),
                        fontsize=6.5, ha="center")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    ax.set_xlabel("C（正则强度倒数，对数刻度）")
    ax.set_ylabel("验证集宏 F1")
    ax.set_title("逻辑回归：正则化参数 C")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_tuning_curves.png")
    plt.close(fig)

def fig_model_comparison():
    rows = []
    repr_df = pd.read_csv(RES_DIR / "stage1_representation.csv")
    base = repr_df[(repr_df.vectorizer == "V0_draft_5k") & (repr_df.stage == "repr_vec")]
    if len(base):
        r = base.iloc[0]
        rows.append(("基线 (draft 5k + 线性SVM)", r["val_accuracy"], r["val_macro_f1"]))
    tune = pd.read_csv(RES_DIR / "stage2_tuning.csv")
    for family in ["SVM", "LinearSVM", "LR"]:
        sub = tune[tune.family == family]
        if len(sub):
            r = sub.sort_values(["val_macro_f1", "val_accuracy"], ascending=False).iloc[0]
            rows.append((f"{family} 最优 {r['params']}", r["val_accuracy"], r["val_macro_f1"]))
    innov_path = RES_DIR / "stage3_innovation.csv"
    if innov_path.exists():
        innov = pd.read_csv(innov_path)
        for _, r in innov.iterrows():
            rows.append((f"创新：{r['family']} + {r['vectorizer']}",
                         r["val_accuracy"], r["val_macro_f1"]))
    labels = [r[0] for r in rows]
    accs = [r[1] for r in rows]
    f1s = [r[2] for r in rows]
    order = np.argsort(f1s)
    labels = [labels[i] for i in order]
    accs = [accs[i] for i in order]
    f1s = [f1s[i] for i in order]

    fig, ax = plt.subplots(figsize=(11.5, max(3.4, 0.42 * len(labels) + 1.4)))
    y = np.arange(len(labels))
    ax.barh(y - 0.2, accs, height=0.4, label="验证准确率", color="#4C72B0")
    ax.barh(y + 0.2, f1s, height=0.4, label="验证宏 F1", color="#55A868")
    ax.set_yticks(y, [l if len(l) < 60 else l[:57] + "..." for l in labels], fontsize=8)
    ax.set_xlim(0.85, 1.0)
    ax.set_xlabel("验证集指标")
    ax.set_title("基线 / 调优 / 创新方案的验证集对比")
    ax.legend(loc="lower right")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_model_comparison.png")
    plt.close(fig)
    return rows

def fig_robustness():
    path = RES_DIR / "stage4_robustness.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if not len(df):
        return
    g = df.groupby("family").agg(mean=("macro_f1_mean", "mean"),
                                 std=("macro_f1_mean", "std")).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(g["family"], g["mean"], yerr=g["std"].fillna(0), capsize=4, color="#4C72B0")
    ax.set_ylim(0.85, 1.0)
    ax.set_title("重复分层交叉验证的宏 F1（均值 ± 标准差，5 折 × 3 种子）")
    ax.set_ylabel("宏 F1")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_robustness.png")
    plt.close(fig)

def fig_loss_curve(prep: str = "raw", vec: str = "V3_word20k_stop",
                   C: float = 10.0, max_iter: int = 100) -> dict:
    set_plot_style()
    train_df, _ = load_data()
    X_all = train_df["text"].astype(str).tolist()
    y_all = train_df["target"].values
    X_tr, X_va, y_tr, _ = train_test_split(
        X_all, y_all, test_size=VAL_SIZE, stratify=y_all, random_state=SEED
    )

    vc = VectorCache(prep, vec)
    A_tr, _ = vc.fit(X_tr, X_va)
    A_tr = A_tr.tocsr()
    y_tr = y_tr.astype(int)
    n, d = A_tr.shape
    n_classes = int(y_tr.max()) + 1
    rows = np.arange(n)

    def stats(theta):
        W = theta[:d * n_classes].reshape(d, n_classes)
        b = theta[d * n_classes:]
        scores = np.asarray(A_tr @ W) + b
        mx = scores.max(axis=1)
        log_norm = mx + np.log(np.exp(scores - mx[:, None]).sum(axis=1))
        losses = log_norm - scores[rows, y_tr]
        ce = float(losses.mean())
        obj = float(0.5 * np.sum(W * W) + C * losses.sum())
        prob = np.exp(scores - log_norm[:, None])
        prob[rows, y_tr] -= 1.0
        grad_w = W + C * (A_tr.T @ prob)
        grad_b = C * prob.sum(axis=0)
        grad = np.concatenate([grad_w.ravel(), grad_b])
        return obj, grad, ce

    def objective(theta):
        obj, grad, _ = stats(theta)
        return obj, grad

    theta0 = np.zeros(d * n_classes + n_classes)
    obj0, _, ce0 = stats(theta0)
    hist_obj = [obj0]
    hist_ce = [ce0]

    def callback(theta):
        obj, _, ce = stats(theta)
        hist_obj.append(obj)
        hist_ce.append(ce)

    result = minimize(
        objective,
        theta0,
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": max_iter, "gtol": 1e-6, "ftol": 1e-12,
                 "maxls": 30, "maxcor": 20},
    )
    final_obj, _, final_ce = stats(result.x)
    if abs(hist_ce[-1] - final_ce) > 1e-12:
        hist_obj.append(final_obj)
        hist_ce.append(final_ce)

    hist_obj = np.asarray(hist_obj, dtype=float)
    hist_ce = np.asarray(hist_ce, dtype=float)

    sklearn_lr = LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                                    random_state=SEED).fit(A_tr, y_tr)
    proba = sklearn_lr.predict_proba(A_tr)
    sklearn_ce = float(-np.log(proba[rows, y_tr]).mean())

    loss_df = pd.DataFrame({
        "iteration": np.arange(len(hist_ce)),
        "objective": hist_obj,
        "avg_cross_entropy": hist_ce,
    })
    loss_df.to_csv(RES_DIR / "loss_curve.csv", index=False, encoding="utf-8-sig")

    summary = {
        "prep": prep,
        "vectorizer": vec,
        "C": float(C),
        "n_train": int(n),
        "n_features": int(d),
        "iterations": int(len(hist_ce) - 1),
        "initial_loss": float(hist_ce[0]),
        "final_loss": float(hist_ce[-1]),
        "best_loss": float(hist_ce.min()),
        "initial_objective": float(hist_obj[0]),
        "final_objective": float(hist_obj[-1]),
        "sklearn_final_loss": sklearn_ce,
        "converged": bool(len(hist_ce) > 10 and abs(hist_ce[-1] - hist_ce[-10]) < 1e-4),
        "optimizer_success": bool(result.success),
        "message": str(result.message),
    }
    (RES_DIR / "loss_curve_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(6.6, 3.25))
    x = np.arange(len(hist_ce))
    ax.plot(x, hist_ce, color="#C44E52", lw=1.8, marker="o",
            markevery=max(1, len(x) // 12), ms=3.2, label="L-BFGS-B 迭代")
    ax.axhline(sklearn_ce, color="#4C72B0", ls="--", lw=1.2,
               label=f"sklearn 最终值 {sklearn_ce:.4f}")
    ax.set_yscale("log")
    ax.set_xlabel("L-BFGS-B 迭代次数")
    ax.set_ylabel("训练平均交叉熵（对数刻度）")
    ax.set_title(f"逻辑回归损失函数收敛曲线（C={C:g}，20k TF-IDF）")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=max(0.03, float(hist_ce.min()) * 0.75),
                top=float(hist_ce.max()) * 1.55)
    ax.annotate(f"初始 {hist_ce[0]:.3f}", xy=(0, hist_ce[0]),
                xytext=(6, 5), textcoords="offset points", fontsize=7)
    ax.annotate(f"最终 {hist_ce[-1]:.4f}", xy=(len(hist_ce) - 1, hist_ce[-1]),
                xytext=(-62, 10), textcoords="offset points", fontsize=7,
                color="#C44E52")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_loss_curve.png")
    plt.close(fig)

    LOG.info("损失曲线：平均交叉熵 %.4f -> %.4f（%d 次迭代；sklearn 终点 %.4f）",
             summary["initial_loss"], summary["final_loss"],
             summary["iterations"], summary["sklearn_final_loss"])
    return summary

def clean_outputs(keep_files) -> None:
    keep = {Path(p).resolve() for p in keep_files}
    if OUT_DIR.exists():
        for path in sorted(OUT_DIR.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_file():
                if path.resolve() not in keep:
                    path.unlink()
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass

    quick_pred = DATA_DIR / "predictions_quick.csv"
    if quick_pred.exists() and quick_pred.resolve() not in keep:
        quick_pred.unlink()

def main(argv=None):
    ap = argparse.ArgumentParser(description="当代人工智能实验一：新闻文本十分类")
    ap.add_argument("--quick", action="store_true", help="快速冒烟测试（缩小搜索网格）")
    args = ap.parse_args(argv)
    global QUICK_MODE
    QUICK_MODE = bool(args.quick)

    tag = time.strftime("%Y%m%d_%H%M%S") + ("_quick" if args.quick else "")
    log_path = setup_logging(tag)
    set_plot_style()
    np.random.seed(SEED)
    LOG.info("实验开始 | seed=%d | sklearn=%s | numpy=%s | pandas=%s | 日志=%s",
             SEED, sklearn.__version__, np.__version__, pd.__version__, log_path.name)

    train_df, test_df = load_data()
    stats = run_eda(train_df, test_df)

    X_all = train_df["text"].astype(str).tolist()
    y_all = train_df["target"].values

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_all, y_all, test_size=VAL_SIZE, stratify=y_all, random_state=SEED
    )
    LOG.info("划分完成：训练 %d / 验证 %d（分层，random_state=%d），无标签测试集 %d 封存",
             len(X_tr), len(X_va), SEED, len(test_df))
    ctx = {
        "eval": {"X_tr": X_tr, "X_va": X_va, "y_tr": y_tr, "y_val": y_va},
        "X_all": X_all, "y_all": y_all,
        "X_test": test_df["text"].astype(str).tolist(),
        "cache": {},
    }

    prep, vec = stage_representation(ctx, args.quick)
    stage_tuning(ctx, prep, vec, args.quick)
    stage_innovation(ctx, prep, None, args.quick)

    cand = []
    for path, stage in ((RES_DIR / "stage2_tuning.csv", "tuning"),
                        (RES_DIR / "stage3_innovation.csv", "innovation")):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["val_macro_f1"].notna()]
        for _, r in df.iterrows():
            cand.append(dict(stage=stage, prep=r["prep"], vectorizer=r["vectorizer"],
                             family=r["family"], params=parse_params(r["params"]),
                             val_accuracy=float(r["val_accuracy"]),
                             val_macro_f1=float(r["val_macro_f1"])))
    cand.sort(key=lambda r: (r["val_macro_f1"], r["val_accuracy"]), reverse=True)
    best_final = cand[0]
    LOG.info("最终选定方案：%s | %s | %s | 验证 acc=%.4f macroF1=%.4f",
             best_final["family"], best_final["prep"], best_final["vectorizer"],
             best_final["val_accuracy"], best_final["val_macro_f1"])

    group_of = {"SVM": "SVM", "LinearSVM": "SVM", "LR": "LR", "NB": "NB", "MLP": "MLP"}
    robust_list, seen_family = [], set()
    for c in cand:
        algo = group_of.get(c["family"])
        if algo is None or algo in seen_family:
            continue
        seen_family.add(algo)
        robust_list.append(c)
    stage_robustness(ctx, robust_list, args.quick)

    diag = stage_error_analysis(ctx, best_final, best_final["prep"], best_final["vectorizer"])
    meta = stage_final(ctx, best_final, best_final["prep"], best_final["vectorizer"])
    meta["data_stats"] = stats
    meta["error_analysis"] = diag
    meta["n_candidate_configs"] = len(cand)

    if args.quick:
        LOG.info("--quick 模式跳过损失曲线生成")
    else:

        import loss_curves

        LOG.info("生成报告用的两张损失曲线 ...")
        loss_curves.main(write_tables=False)

    keep = {FIG_DIR / "fig_loss_svm.png", FIG_DIR / "fig_loss_lr.png",
            DATA_DIR / "predictions.csv"}
    clean_outputs(keep)
    if QUICK_MODE:
        LOG.info("实验结束（quick）。仅保留两张损失图；未写预测文件")
    else:
        LOG.info("实验结束。仅保留两张损失图与 predictions.csv；中间文件已清理")
    return meta

if __name__ == "__main__":
    main()
