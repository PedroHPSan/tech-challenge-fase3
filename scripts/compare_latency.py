"""Compara a latência de inferência: pipeline scikit-learn (joblib) x ONNX Runtime.

Mede só o modelo, no mesmo processo e com os mesmos pacientes (1 por chamada, como
na API; texto e sinais clínicos do dataset). Uso: python -m scripts.compare_latency --n 2000
"""

import argparse
import statistics
import time
from pathlib import Path

import joblib

from src.config import settings
from src.data import load_dataset
from src.features import usable_features
from src.train import load_onnx, predict_onnx


def measure(predict, rows: list, n: int) -> list[float]:
    for row in rows:  # aquecimento
        predict(row)
    times = []
    for i in range(n):
        start = time.perf_counter()
        predict(rows[i % len(rows)])
        times.append((time.perf_counter() - start) * 1000)
    return sorted(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args()

    model_path = Path(settings.model_path)
    onnx_path = model_path.with_suffix(".onnx")
    pipe = joblib.load(model_path)
    session = load_onnx(str(onnx_path))
    df = load_dataset(settings.data_path).drop_duplicates("text").reset_index(drop=True)
    features = [c for c in usable_features(df) if c in {i.name for i in session.get_inputs()}]
    # Um paciente por chamada; as entradas ficam prontas antes de cronometrar.
    rows = [df.iloc[[i]] for i in range(len(df))]
    onnx_rows = [([r["text"].iloc[0]], {c: [r[c].iloc[0]] for c in features}) for r in rows]

    results = {
        "scikit-learn": measure(pipe.predict_proba, rows, args.n),
        "ONNX Runtime": measure(lambda args_: predict_onnx(session, *args_), onnx_rows, args.n),
    }

    print(f"{args.n} predições de 1 texto cada (ms)")
    print(f"{'modelo':<14}{'média':>8}{'p50':>8}{'p95':>8}{'máx':>8}")
    for name, t in results.items():
        p95 = t[int(len(t) * 0.95) - 1]
        print(f"{name:<14}{statistics.mean(t):>8.3f}{t[len(t) // 2]:>8.3f}{p95:>8.3f}{t[-1]:>8.3f}")
    speedup = statistics.mean(results["scikit-learn"]) / statistics.mean(results["ONNX Runtime"])
    print(f"ganho na média: {speedup:.1f}x")

    labels, _ = predict_onnx(session, list(df["text"]), {c: list(df[c]) for c in features})
    agree = (labels == pipe.predict(df)).mean()
    print(f"concordância de rótulos (pacientes do dataset): {agree:.0%}")
    size_kb = {p.suffix: p.stat().st_size / 1024 for p in (model_path, onnx_path)}
    print(f"tamanho: joblib {size_kb['.joblib']:.0f} KB, onnx {size_kb['.onnx']:.0f} KB")


if __name__ == "__main__":
    main()
