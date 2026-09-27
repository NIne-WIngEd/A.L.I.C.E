#!/usr/bin/env python3
"""Inventory an actual two-GPU runtime without loading N0 data or training.

Run once with Python for a single-process inventory. To prove two-rank NCCL
plumbing, run: torchrun --standalone --nproc_per_node=2 THIS_FILE --collective
The output is infrastructure evidence only; it never grants P43 authority.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from typing import Any


PACKAGES=("torch", "accelerate", "transformers", "tokenizers", "safetensors",
          "datasets", "huggingface-hub", "numpy")
OLD_PROJECTION_BYTES=17_754_461_984
MAX_FRACTION=0.85


def packages() -> dict[str, str | None]:
    result={}
    for package in PACKAGES:
        try:
            result[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package]=None
    return result


def driver() -> list[dict[str, str]] | None:
    if not shutil.which("nvidia-smi"):
        return None
    command=["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version",
             "--format=csv,noheader,nounits"]
    try:
        result=subprocess.run(command,check=True,capture_output=True,text=True,timeout=15)
    except (OSError,subprocess.CalledProcessError,subprocess.TimeoutExpired):
        return None
    values=[]
    for line in result.stdout.splitlines():
        parts=[part.strip() for part in line.split(",")]
        if len(parts)!=5:
            raise RuntimeError("unexpected nvidia-smi GPU inventory format")
        values.append(dict(zip(("index","name","uuid","memory_mib","driver_version"),parts)))
    return values


def inventory() -> tuple[dict[str, Any], Any]:
    result: dict[str,Any]={
        "schema":"alice.n0.two-gpu-runtime-inventory.v1",
        "authority":"HARDWARE_ONLY_NO_N0_DATA_NO_TRAINING",
        "python":platform.python_version(),
        "machine":platform.machine(),
        "glibc":platform.libc_ver(),
        "packages":packages(),
        "driver_devices":driver(),
        "old_p43_projection_bytes":OLD_PROJECTION_BYTES,
        "current_route_memory_measured":False,
        "gpu_training_authorized":False,
    }
    try:
        import torch
    except ImportError:
        result["status"]="NO_TORCH"
        return result,None
    result["torch_build"]=torch.__version__
    result["torch_cuda_build"]=torch.version.cuda
    result["torch_cuda_available"]=torch.cuda.is_available()
    result["nccl_available"]=bool(torch.distributed.is_nccl_available())
    if result["nccl_available"]:
        result["nccl_version"]=torch.cuda.nccl.version()
    devices=[]
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props=torch.cuda.get_device_properties(index)
            total=int(props.total_memory)
            devices.append({
                "index":index,"name":props.name,
                "total_memory_bytes":total,
                "compute_capability":list(torch.cuda.get_device_capability(index)),
                "old_projection_below_85_percent":OLD_PROJECTION_BYTES<=MAX_FRACTION*total,
            })
    result["devices"]=devices
    result["status"]="INVENTORIED_NOT_N0_AUTHORITY"
    return result,torch


def collective(result: dict[str,Any],torch: Any) -> dict[str,Any]:
    if torch is None or not result.get("nccl_available"):
        raise RuntimeError("two-rank NCCL is unavailable")
    import torch.distributed as dist
    if int(os.environ.get("WORLD_SIZE","0"))!=2:
        raise RuntimeError("run exactly two processes with torchrun")
    rank=int(os.environ["RANK"])
    local_rank=int(os.environ["LOCAL_RANK"])
    if len(result["devices"])!=2 or local_rank not in (0,1):
        raise RuntimeError("exactly two visible GPUs and one distinct GPU per rank required")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",device_id=torch.device("cuda",local_rank))
    try:
        probe=torch.tensor([rank+1.0],device="cuda")
        dist.all_reduce(probe)
        torch.cuda.synchronize(local_rank)
        if float(probe.item())!=3.0:
            raise RuntimeError("NCCL all-reduce returned an unexpected value")
        local={"rank":rank,"local_rank":local_rank,
               "device":result["devices"][local_rank]}
        gathered=[None,None]
        dist.all_gather_object(gathered,local)
        if sorted(item["local_rank"] for item in gathered)!=[0,1]:
            raise RuntimeError("ranks did not use distinct GPUs")
        result["collective"]="TWO_RANK_NCCL_ALLREDUCE_PASS"
        result["ranks"]=gathered
    finally:
        dist.destroy_process_group()
    return result


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collective",action="store_true")
    args=parser.parse_args()
    result,torch=inventory()
    if args.collective:
        try:
            result=collective(result,torch)
        except Exception as exc:
            result["status"]="COLLECTIVE_FAILED_NOT_N0_AUTHORITY"
            result["collective_error"]=f"{type(exc).__name__}: {exc}"
            print(json.dumps(result,sort_keys=True,indent=2),flush=True)
            return 3
        if int(os.environ["RANK"])!=0:
            return 0
    print(json.dumps(result,sort_keys=True,indent=2),flush=True)
    return 0 if result["status"]=="INVENTORIED_NOT_N0_AUTHORITY" else 2


if __name__=="__main__":
    raise SystemExit(main())
