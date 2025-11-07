import re
import subprocess
import tempfile
from pathlib import Path

BASE_DIR = Path('/Users/christian/Documents/Projects/trading/3StateMachineEngine')
SOURCE = BASE_DIR / 'dualday.py'


def read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding='utf-8')


def replace_param(src: str, name: str, value_repr: str) -> str:
    # Replace lines like: NAME = <value>
    pattern = rf"^(\s*{re.escape(name)}\s*=\s*).*$"
    return re.sub(pattern, rf"\g<1>{value_repr}", src, flags=re.MULTILINE)


def run_python(path: Path) -> str:
    env_python = BASE_DIR / 'env' / 'bin' / 'python3'
    python = str(env_python if env_python.exists() else 'python3')
    completed = subprocess.run([python, str(path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(BASE_DIR))
    return completed.stdout


def extract_metrics(output: str) -> str:
    # Grab Holdout metrics blocks
    lines = output.splitlines()
    keep = []
    tags = ("Holdout (Paper):", "Holdout (Real):")
    for i, line in enumerate(lines):
        if any(tag in line for tag in tags):
            # Skip Baseline blocks; keep only the NEW 3-state dual-day metrics
            if line.strip().startswith("Baseline"):
                continue
            block = lines[i:i+8]
            keep.extend(block)
    return "\n".join(keep) if keep else "<No Holdout metrics found>"


def make_variant(replacements: dict, label: str) -> str:
    src = read_text(SOURCE)
    for k, v in replacements.items():
        src = replace_param(src, k, v)
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / f'dualday_{label}.py'
        write_text(tmp_path, src)
        out = run_python(tmp_path)
        return out


def main():
    print("\n================ DEFAULTS (3-state dual-day) ================")
    baseline_out = make_variant({}, 'baseline')
    print(extract_metrics(baseline_out))

    tests = [
        ("VOL_TH_3X=0.30", {"VOL_TH_3X": "0.30"}),
        ("VOL_TH_3X=0.40", {"VOL_TH_3X": "0.40"}),
        ("DD_TH_2X=0.15",  {"DD_TH_2X":  "0.15"}),
        ("DD_TH_2X=0.25",  {"DD_TH_2X":  "0.25"}),
        ("SMA_SLOW=150",   {"SMA_SLOW":   "150"}),
        ("SMA_SLOW=250",   {"SMA_SLOW":   "250"}),
        ("MID_UP_WINDOW_WEEKS_BENIGN=2", {"MID_UP_WINDOW_WEEKS_BENIGN": "2"}),
        ("MID_UP_WINDOW_WEEKS_BENIGN=8", {"MID_UP_WINDOW_WEEKS_BENIGN": "8"}),
    ]

    for label, repl in tests:
        print(f"\n================ {label} ================")
        out = make_variant(repl, label.replace('=', '_'))
        print(extract_metrics(out))


if __name__ == '__main__':
    main()


