from pathlib import Path

from selection_bench.data import load_data
from selection_bench.protocol import build_target, run_target, write_results


ROOT = Path(__file__).resolve().parent


def main() -> None:
    data = load_data(ROOT / "data" / "openglt_selection.npz")
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    results = [run_target(build_target(data, name)) for name in ("MedQ", "P95")]
    write_results(results, output)


if __name__ == "__main__":
    main()
