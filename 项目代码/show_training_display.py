"""Display source code and execute the actual training commands."""
import json
import os
from pathlib import Path
import subprocess
import sys

project = Path(__file__).resolve().parent
print("=== train_classifier.py 关键训练代码 ===", flush=True)
lines = (project / "train_classifier.py").read_text(encoding="utf-8-sig").splitlines()
for number, line in enumerate(lines[48:65], 49):
    print(f"{number:3}  {line}", flush=True)

env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
for script, args in (("train_classifier.py", ["notices.jsonl"]),
                     ("extend_training.py", [])):
    print(f"\n> python {script} {' '.join(args)}", flush=True)
    result = subprocess.run([sys.executable, "-W", "ignore", script, *args],
                            cwd=project, env=env)
    print(f"实际退出码  {result.returncode}", flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)

print("\n=== 训练集三折交叉验证结果 ===", flush=True)
metrics = json.loads((project / "extended_cv.json").read_text(encoding="utf-8"))
for name, score in metrics.items():
    print(f"{name:24} {score['correct']}/30  "
          f"accuracy={score['accuracy']:.3f}  macro_F1={score['macro_f1']:.3f}")
print("\n训练运行完成。15条独立测试与30条训练集交叉验证分别列出。")

