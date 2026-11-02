"""Mede a latência da API (baseline local ou em container).

Uso: python -m scripts.benchmark_latency --url http://localhost:8000 --n 200
"""

import argparse
import statistics
import time

import httpx

TEXTS = [
    "chest pain",
    "fever",
    "ankle sprain",
    "abdominal pain, acute",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    times: list[float] = []
    with httpx.Client(base_url=args.url, timeout=10) as client:
        for text in TEXTS * 3:  # aquecimento
            client.post("/predict", json={"text": text})
        for i in range(args.n):
            start = time.perf_counter()
            r = client.post("/predict", json={"text": TEXTS[i % len(TEXTS)]})
            r.raise_for_status()
            times.append((time.perf_counter() - start) * 1000)

    times.sort()
    print(f"requisições: {len(times)}")
    print(f"média: {statistics.mean(times):.2f} ms")
    print(f"p50:   {times[len(times) // 2]:.2f} ms")
    print(f"p95:   {times[int(len(times) * 0.95) - 1]:.2f} ms")
    print(f"máx:   {times[-1]:.2f} ms")


if __name__ == "__main__":
    main()
