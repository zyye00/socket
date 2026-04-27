#!/usr/bin/env python3

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import zmq

from message import Result


class Orca:
    def __init__(self, elements):
        self.elements = tuple(elements)
        orca_exe = shutil.which("orca")
        if orca_exe is None:
            raise FileNotFoundError("Could not find ORCA executable in PATH")
        self.orca_exe = os.path.realpath(orca_exe)

    def run(self, task):
        tmp = tempfile.mkdtemp(prefix=f"orca-task-{task.id}-")
        try:
            inp = os.path.join(tmp, "job.inp")
            out = os.path.join(tmp, "job.out")
            engrad = os.path.join(tmp, "job.engrad")

            with open(inp, "w") as f:
                f.write("!B3LYP 6-311G** Engrad\n")
                f.write("%pal\n")
                f.write("  nprocs 4\n")
                f.write("end\n")
                f.write("%MaxCore 4000\n")
                f.write("* xyz 0 1\n")
                for element, (x, y, z) in zip(self.elements, task.coords):
                    f.write(f"{element} {x} {y} {z}\n")
                f.write("*\n")

            with open(out, "w") as f:
                subprocess.run(
                    [self.orca_exe, inp],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True
                )

            if not os.path.exists(engrad):
                raise RuntimeError(
                    f"ORCA did not produce {engrad}; see {out} for details"
                )

            with open(engrad) as f:
                lines = [line.strip() for line in f if "#" not in line and line.strip()]
        except Exception:
            print(
                f"Task {task.id}: preserving ORCA work directory for debugging: {tmp}"
            )
            raise
        else:
            shutil.rmtree(tmp)

        n_atoms, energy = int(lines[0]), float(lines[1])
        gradient = np.array(lines[2 : 2 + 3 * n_atoms], dtype=float).reshape(-1, 3)

        return Result(task.id, energy=energy, forces=-gradient)


def get_driver_id():
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = os.environ.get("SLURM_JOB_ID")
    proc_id = os.environ.get("SLURM_PROCID")

    if array_job_id and array_task_id:
        return f"driver-{array_job_id}_{array_task_id}"
    if job_id and proc_id:
        return f"driver-{job_id}_{proc_id}"
    if job_id:
        return f"driver-{job_id}"
    return f"driver-{os.getpid()}"


def main():
    driver_id = get_driver_id()
    print(f"Driver {driver_id} starting...")

    server_info_path = Path("server_info.json")
    print(f"Driver {driver_id}: waiting for server_info.json...")
    while True:
        if not server_info_path.exists():
            time.sleep(1)
            continue

        try:
            with server_info_path.open(encoding="utf-8") as f:
                server_info = json.load(f)
            break
        except json.JSONDecodeError:
            time.sleep(0.2)

    host = server_info["host"]
    port = int(server_info["port"])
    elements = server_info["elements"]

    context = zmq.Context.instance()
    driver_socket = context.socket(zmq.DEALER)
    driver_socket.setsockopt(zmq.IDENTITY, str(driver_id).encode())
    orca = Orca(elements)
    print(f"Driver {driver_id}: using ORCA executable {orca.orca_exe} ")

    try:
        print(f"Driver {driver_id} connecting to server at {host}:{port}...")
        driver_socket.connect(f"tcp://{host}:{port}")
        print(f"Driver {driver_id} connected")

        driver_socket.send_pyobj({"type": "ready"})

        while True:
            task = driver_socket.recv_pyobj()

            if task is None:
                print(f"Driver {driver_id}: no more tasks, exiting")
                break

            print(f"Driver {driver_id} received task {task.id}")

            result = orca.run(task)

            driver_socket.send_pyobj(result)
            print(f"Driver {driver_id} sent result for task {task.id}")
    finally:
        driver_socket.close()
        context.term()


if __name__ == "__main__":
    main()
