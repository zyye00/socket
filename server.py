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
    # Use SLURM_NODELIST if available
    nodelist = os.environ.get("SLURM_NODELIST")
    if nodelist:
        first_node = nodelist.split(",")[0].split("[")[0]
        return first_node

    # Fallback to hostname
    return socket.gethostname()


def schedule_next_task(server_socket, identity, tasks, next_task_idx):
    if next_task_idx < len(tasks):
        task = tasks[next_task_idx]
        server_socket.send_multipart([identity, pickle.dumps(task)])
        print(f"Task {task.id} sent to {identity.decode(errors='ignore')}")
        return next_task_idx + 1

    server_socket.send_multipart([identity, pickle.dumps(None)])
    return next_task_idx


def main():
    server_info_path = "server_info.json"
    server_info_tmp_path = f"{server_info_path}.tmp"

    num_tasks = 8
    pending_tasks = num_tasks
    coords = np.array([
        [0.0000, 0.0000, 0.0626],
        [-0.7920, 0.0000, -0.4973],
        [0.7920, 0.0000, -0.4973]
    ])
    displacement = np.array([[0, 0.01, 0],
                             [0, 0.00, 0],
                             [0, 0.00, 0]])
    tasks = [Task(id=i, coords=coords + displacement * i) for i in range(num_tasks)]
    next_task_idx = 0
    active_drivers = set()

    print(f"Created {num_tasks} tasks")

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

        print(f"Waiting for {num_tasks} tasks to complete...\n")

        start = time.time()
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
                continue

            if isinstance(message, Result):
                results[message.id] = message
                print(
                    f"[RESULT] Task {message.id} from {driver_name}: "
                    f"E={message.energy:.6f}"
                )

                pending_tasks -= 1
                next_task_idx = schedule_next_task(
                    server_socket, identity, tasks, next_task_idx
                )
                continue

            print(f"[ERROR] Unexpected message from {driver_name}: {type(message)}")

        end = time.time()
        print(f"\n[✓] All {num_tasks} tasks completed!")
        print(f"Total time: {end - start:.2f} seconds")
        print("Results summary:")
        for task_id in sorted(results.keys()):
            energy = results[task_id].energy
            print(f"  Task {task_id}: E={energy:.6f}")
    finally:
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
