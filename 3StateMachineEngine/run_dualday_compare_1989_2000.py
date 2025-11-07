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
    pattern = rf"^(\s*{re.escape(name)}\s*=\s*).*$"
    return re.sub(pattern, rf"\g<1>{value_repr}", src, flags=re.MULTILINE)


def run_python(path: Path) -> str:
    env_python = BASE_DIR / 'env' / 'bin' / 'python3'
    python = str(env_python if env_python.exists() else 'python3')
    completed = subprocess.run([python, str(path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(BASE_DIR))
    return completed.stdout


def extract_holdout_metrics(output: str) -> str:
    lines = output.splitlines()
    keep = []
    tags = ("Holdout (Paper):", "Holdout (Real):")
    for i, line in enumerate(lines):
        if any(tag in line for tag in tags):
            block = lines[i:i+8]
            if any('Baseline' in b for b in block):
                continue
            keep.extend(block)
    return "\n".join(keep) if keep else "<No Holdout metrics found>"


def make_variant(sma_value: int) -> str:
    src = read_text(SOURCE)
    src = replace_param(src, 'DATA_START', '"1988-01-01"')
    src = replace_param(src, 'SMA_SLOW', str(sma_value))
    src = replace_param(src, 'HOLDOUT', '("1989-01-01", "2000-01-01")')
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / f'dualday_sma_{sma_value}.py'
        write_text(tmp_path, src)
        out = run_python(tmp_path)
        return extract_holdout_metrics(out)


def main():
    print("\n===== Dual-Day 3-State | HOLDOUT 1989-01-01 → 2000-01-01 =====\n")
    for sma in (150, 200):
        print(f"--- SMA_SLOW = {sma} ---")
        res = make_variant(sma)
        print(res)
        print()


if __name__ == '__main__':
    main()


