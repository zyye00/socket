#!/usr/bin/env python3

import contextlib
import json
import os
import pickle
import socket
import time

import numpy as np
import zmq

from message import Result, Task

results = {}
ELEMENTS = ["O", "H", "H"]


def get_bind_host():
    slurmd_nodename = os.environ.get("SLURMD_NODENAME")
    if slurmd_nodename:
        return slurmd_nodename

    for env_name in ("SLURM_NODELIST", "SLURM_JOB_NODELIST"):
        nodelist = os.environ.get(env_name)
        if not nodelist:
            continue
        if "[" not in nodelist:
            return nodelist.split(",")[0]

    return socket.gethostname()


def schedule_next_task(server_socket, identity, tasks, next_task_idx):
    if next_task_idx is None:
        return None

    if next_task_idx < len(tasks):
        task = tasks[next_task_idx]
        server_socket.send_multipart([identity, pickle.dumps(task)])
        print(f"Task {task.id} sent to {identity.decode(errors='ignore')}")
        return next_task_idx + 1

    return next_task_idx


def main():
    server_info_path = "server_info.json"
    server_info_tmp_path = f"{server_info_path}.tmp"

    num_tasks = 8
    num_rounds = 2
    coords = np.array([
        [0.0000, 0.0000, 0.0626],
        [-0.7920, 0.0000, -0.4973],
        [0.7920, 0.0000, -0.4973]
    ])
    displacement = np.array([[0, 0.01, 0],
                             [0, 0.00, 0],
                             [0, 0.00, 0]])
    active_drivers = set()

    bind_host = get_bind_host()

    context = zmq.Context.instance()
    server_socket = context.socket(zmq.ROUTER)

    try:
        server_socket.bind("tcp://*:0")
        endpoint = server_socket.getsockopt_string(zmq.LAST_ENDPOINT)
        actual_port = int(endpoint.rsplit(":", 1)[1])

        server_info = {
            "host": bind_host,
            "port": actual_port,
            "elements": ELEMENTS,
        }
        with open(server_info_tmp_path, "w", encoding="utf-8") as f:
            json.dump(server_info, f)
        os.replace(server_info_tmp_path, server_info_path)
        print(
            "Server info saved to server_info.json: "
            f"{bind_host}:{actual_port}, elements={ELEMENTS}"
        )
        print(f"Server listening on {endpoint}")

        total_start = time.time()
        for round_idx in range(num_rounds):
            round_results = {}
            pending_tasks = num_tasks
            next_task_idx = 0
            multiplier = 1 if round_idx == 0 else 2
            tasks = [
                Task(id=i, coords=coords + displacement * i * multiplier)
                for i in range(num_tasks)
            ]

            print(f"Created {num_tasks} tasks for round {round_idx}/{num_rounds} ")
            print(
                f"Waiting for round {round_idx}/{num_rounds} "
                f"to complete all {num_tasks} tasks...\n"
            )

            if round_idx > 0 and active_drivers:
                print(
                    f"Dispatching initial round {round_idx} tasks to "
                    f"{len(active_drivers)} active drivers"
                )
                for identity in active_drivers:
                    next_task_idx = schedule_next_task(
                        server_socket, identity, tasks, next_task_idx
                    )

            round_start = time.time()
            while pending_tasks > 0:
                identity, data = server_socket.recv_multipart()
                message = pickle.loads(data)
                driver_name = identity.decode(errors="ignore")

                if identity not in active_drivers:
                    active_drivers.add(identity)
                    print(f"[+] Driver connected: {driver_name}")

                if isinstance(message, dict) and message.get("type") == "ready":
                    next_task_idx = schedule_next_task(
                        server_socket, identity, tasks, next_task_idx
                    )

                elif isinstance(message, Result):
                    round_results[message.id] = message
                    results[(round_idx, message.id)] = message
                    print(
                        f"[ROUND {round_idx}] [RESULT] Task {message.id} "
                        f"from {driver_name}: E={message.energy:.6f}"
                    )

                    pending_tasks -= 1
                    next_task_idx = schedule_next_task(
                        server_socket, identity, tasks, next_task_idx
                    )

                else:
                    print(
                        f"[ERROR] Unexpected message from {driver_name}: "
                        f"{type(message)}"
                    )

            round_end = time.time()
            print(f"\n[ROUND {round_idx}] All {num_tasks} tasks completed!")
            print(
                f"[ROUND {round_idx}] Total time: "
                f"{round_end - round_start:.2f} seconds"
            )
            print(f"[ROUND {round_idx}] Results summary:")
            for task_id in sorted(round_results.keys()):
                energy = round_results[task_id].energy
                print(f"  Task {task_id}: E={energy:.6f}")
            print()

        total_end = time.time()
        print(f"All {num_rounds} rounds completed!")
        print(f"Total time: {total_end - total_start:.2f} seconds")
    finally:
        for driver in active_drivers:
            server_socket.send_multipart([driver, pickle.dumps(None)])
        server_socket.close()
        context.term()
        try:
            os.remove(server_info_path)
            print("Removed server_info.json")
        except FileNotFoundError:
            pass
        with contextlib.suppress(FileNotFoundError):
            os.remove(server_info_tmp_path)


if __name__ == "__main__":
    main()
