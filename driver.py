#!/usr/bin/env python3

import os
import time
from pathlib import Path

import numpy as np
import zmq

from message import Result


def run_orca(task):
    # with tempfile.TemporaryDirectory() as tmp:
    #     inp = os.path.join(tmp, "job.inp")
    #     out = os.path.join(tmp, "job.out")
    #
    #     with open(inp, "w") as f:
    #         f.write("! B3LYP def2-SVP\n")
    #         f.write("* xyz 0 1\n")
    #         for x, y, z in coords:
    #             f.write(f"H {x} {y} {z}\n")
    #         f.write("*\n")
    #
    #     subprocess.run(
    #         ["orca", inp],
    #         stdout=open(out, "w"),
    #         stderr=subprocess.STDOUT,
    #         check=True
    #     )

    time.sleep(0.2)
    return Result(task.id, energy=-1.0, forces=np.zeros_like(task.coords))


def main():
    driver_id = os.environ.get("SLURM_PROCID", f"driver-{os.getpid()}")
    print(f"Driver {driver_id} starting...")

    server_info_path = Path("server_info.txt")
    print(f"Driver {driver_id}: waiting for server_info.txt...")
    while not server_info_path.exists():
        time.sleep(1)

    with server_info_path.open() as f:
        host = f.readline().strip()
        port = int(f.readline().strip())
    print(f"Driver {driver_id}: read server info from server_info.txt: {host}:{port}")

    context = zmq.Context.instance()
    driver_socket = context.socket(zmq.DEALER)
    driver_socket.setsockopt(zmq.IDENTITY, str(driver_id).encode())

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

            result = run_orca(task)

            driver_socket.send_pyobj(result)
            print(f"Driver {driver_id} sent result for task {task.id}")
    finally:
        driver_socket.close()
        context.term()


if __name__ == "__main__":
    main()
