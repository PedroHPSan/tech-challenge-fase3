"""Compara a latência de inferência: pipeline scikit-learn (joblib) x ONNX Runtime.

Mede só o modelo, no mesmo processo e com os mesmos textos (1 texto por chamada,
como na API). Uso: python -m scripts.compare_latency --n 2000
"""

import argparse
import statistics
import time
from pathlib import Path

import joblib

from src.config import settings
from src.data import load_dataset
from src.train import load_onnx, predict_onnx


def measure(predict, texts: list[str], n: int) -> list[float]:
    for text in texts:  # aquecimento
        predict(text)
    times = []
    for i in range(n):
        start = time.perf_counter()
        predict(texts[i % len(texts)])
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
    texts = list(load_dataset(settings.data_path)["text"].unique())

    results = {
        "scikit-learn": measure(lambda t: pipe.predict_proba([t]), texts, args.n),
        "ONNX Runtime": measure(lambda t: predict_onnx(session, [t]), texts, args.n),
    }

    print(f"{args.n} predições de 1 texto cada (ms)")
    print(f"{'modelo':<14}{'média':>8}{'p50':>8}{'p95':>8}{'máx':>8}")
    for name, t in results.items():
        p95 = t[int(len(t) * 0.95) - 1]
        print(f"{name:<14}{statistics.mean(t):>8.3f}{t[len(t) // 2]:>8.3f}{p95:>8.3f}{t[-1]:>8.3f}")
    speedup = statistics.mean(results["scikit-learn"]) / statistics.mean(results["ONNX Runtime"])
    print(f"ganho na média: {speedup:.1f}x")

    labels, _ = predict_onnx(session, texts)
    agree = (labels == pipe.predict(texts)).mean()
    print(f"concordância de rótulos (frases do dataset): {agree:.0%}")
    size_kb = {p.suffix: p.stat().st_size / 1024 for p in (model_path, onnx_path)}
    print(f"tamanho: joblib {size_kb['.joblib']:.0f} KB, onnx {size_kb['.onnx']:.0f} KB")


if __name__ == "__main__":
    main()
