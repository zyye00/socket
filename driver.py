#!/usr/bin/env python3

import os
import socket

import numpy as np

from message import ConnectionClosedError, Result, recv_message, send_message


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

    return Result(task.id, energy=-1.0, forces=np.zeros_like(task.coords))


def main():
    driver_id = os.environ.get("SLURM_PROCID", f"driver-{os.getpid()}")
    print(f"Driver {driver_id} starting...")

    with open("server_info.txt") as f:
        host = f.readline().strip()
        port = int(f.readline().strip())
    print(f"Read server info from server_info.txt: {host}:{port}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"Driver {driver_id} connecting to server at {host}:{port}...")
        s.connect((host, port))
        print(f"Driver {driver_id} connected")

        while True:
            try:
                task = recv_message(s)
            except ConnectionClosedError:
                print(f"Driver {driver_id}: server closed connection, exiting")
                break
            print(f"Driver {driver_id} received task {task.id}")

            result = run_orca(task)

            send_message(s, result)
            print(f"Driver {driver_id} sent result for task {task.id}")


if __name__ == "__main__":
    main()
