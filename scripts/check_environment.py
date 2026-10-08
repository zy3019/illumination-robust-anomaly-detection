"""检查环境并执行真实 CPU/GPU 运算；从项目任意位置运行均可。"""

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import torch
import torchvision


def command_output(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    # Windows 不一定安装 IANA 时区数据，固定 UTC+8 作为回退。
    try:
        zone = ZoneInfo("Asia/Shanghai")
    except ZoneInfoNotFoundError:
        from datetime import timedelta, timezone
        zone = timezone(timedelta(hours=8))
    report = {
        "checked_at": datetime.now(zone).isoformat(),
        "os": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "git": command_output(["git", "--version"]),
        "gpu_driver": command_output([
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]),
    }
    failures = []
    devices = ["cpu"] + (["cuda"] if report["cuda_available"] else [])
    for device in devices:
        try:
            a = torch.tensor([[1., 2.], [3., 4.]], device=device)
            result = a @ a
            torch.testing.assert_close(
                result.cpu(), torch.tensor([[7., 10.], [15., 22.]])
            )
            w = torch.tensor(2., device=device, requires_grad=True)
            loss = (3 * w - 1).square()
            loss.backward()
            torch.testing.assert_close(w.grad.cpu(), torch.tensor(30.))
            if device == "cuda":
                torch.cuda.synchronize()
                props = torch.cuda.get_device_properties(0)
                report["gpu_name"] = props.name
                report["gpu_memory_bytes"] = props.total_memory
            report[f"{device}_matrix_and_autograd"] = "passed"
        except Exception as exc:
            report[f"{device}_matrix_and_autograd"] = f"failed: {exc}"
            failures.append(device)
    if args.require_cuda and not report["cuda_available"]:
        failures.append("CUDA required but unavailable")
    report["verification_passed"] = not failures
    path = Path(__file__).resolve().parents[1] / "records" / "environment.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
