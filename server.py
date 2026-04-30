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


def create_round_tasks(coords, displacement, num_tasks, round_idx):
    multiplier = 1 if round_idx == 0 else 2
    tasks = [
        Task(id=i, coords=coords + displacement * i * multiplier)
        for i in range(num_tasks)
    ]
    return tasks, multiplier


def schedule_next_task(server_socket, identity, tasks, next_task_idx):
    if next_task_idx >= len(tasks):
        return next_task_idx

    task = tasks[next_task_idx]
    server_socket.send_multipart([identity, pickle.dumps(task)])
    print(f"Task {task.id} sent to {identity.decode(errors='ignore')}")
    return next_task_idx + 1


def dispatch_idle_drivers(server_socket, idle_drivers, tasks, next_task_idx):
    while idle_drivers and next_task_idx < len(tasks):
        identity = idle_drivers.pop()
        next_task_idx = schedule_next_task(
            server_socket, identity, tasks, next_task_idx
        )
    return next_task_idx


def handle_driver_message(
    identity,
    message,
    active_drivers,
    idle_drivers,
    round_results,
    round_number,
):
    driver_name = identity.decode(errors="ignore")

    if identity not in active_drivers:
        active_drivers.add(identity)
        print(f"[+] Driver connected: {driver_name}")

    if isinstance(message, dict) and message.get("type") == "ready":
        idle_drivers.add(identity)
        return 0

    if isinstance(message, Result):
        round_results[message.id] = message
        results[(round_number, message.id)] = message
        print(
            f"[ROUND {round_number}] [RESULT] Task {message.id} "
            f"from {driver_name}: E={message.energy:.6f}"
        )
        idle_drivers.add(identity)
        return 1

    print(f"[ERROR] Unexpected message from {driver_name}: {type(message)}")
    return 0


def run_round(
    server_socket,
    tasks,
    num_tasks,
    round_number,
    active_drivers,
    idle_drivers,
):
    round_results = {}
    pending_tasks = num_tasks
    next_task_idx = 0
    round_start = time.time()

    while pending_tasks > 0:
        next_task_idx = dispatch_idle_drivers(
            server_socket, idle_drivers, tasks, next_task_idx
        )

        identity, data = server_socket.recv_multipart()
        message = pickle.loads(data)
        completed_tasks = handle_driver_message(
            identity,
            message,
            active_drivers,
            idle_drivers,
            round_results,
            round_number,
        )
        pending_tasks -= completed_tasks

    round_end = time.time()
    return round_results, round_end - round_start


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
    idle_drivers = set()

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
            round_number = round_idx + 1
            tasks, multiplier = create_round_tasks(
                coords, displacement, num_tasks, round_idx
            )

            print(
                f"Created {num_tasks} tasks for round {round_number}/{num_rounds} "
                f"(multiplier={multiplier})"
            )
            print(
                f"Waiting for round {round_number}/{num_rounds} "
                f"to complete all {num_tasks} tasks...\n"
            )

            round_results, round_elapsed = run_round(
                server_socket,
                tasks,
                num_tasks,
                round_number,
                active_drivers,
                idle_drivers,
            )

            print(f"\n[ROUND {round_number}] All {num_tasks} tasks completed!")
            print(f"[ROUND {round_number}] Total time: {round_elapsed:.2f} seconds")
            print(f"[ROUND {round_number}] Results summary:")
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
