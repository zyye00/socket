#!/usr/bin/env python3

import os
import pickle
import socket

import numpy as np
import zmq

from message import Result, Task

results = {}


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
    server_info_path = "server_info.txt"

    num_tasks = 10
    pending_tasks = num_tasks
    tasks = [Task(id=i, coords=np.zeros((2, 3))) for i in range(num_tasks)]
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

        with open(server_info_path, "w") as f:
            f.write(f"{bind_host}\n{actual_port}\n")
        print(f"Server info saved to server_info.txt: {bind_host}:{actual_port}")

        print(f"Waiting for {num_tasks} tasks to complete...\n")

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

        print(f"\n[✓] All {num_tasks} tasks completed!")
        print("Results summary:")
        for task_id in sorted(results.keys()):
            energy = results[task_id].energy
            print(f"  Task {task_id}: E={energy:.6f}")
    finally:
        server_socket.close()
        context.term()
        try:
            os.remove(server_info_path)
            print("Removed server_info.txt")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
