#!/usr/bin/env python3

import os
import socket
import threading
from queue import Empty, Queue
from threading import Lock

import numpy as np

from message import Task, recv_message, send_message

task_queue = Queue()
results = {}
results_lock = Lock()
pending_tasks = 0
pending_lock = Lock()
all_tasks_done = threading.Event()


def get_bind_host():
    # Use SLURM_NODELIST if available
    nodelist = os.environ.get("SLURM_NODELIST")
    if nodelist:
        first_node = nodelist.split(",")[0].split("[")[0]
        return first_node

    # Fallback to hostname
    return socket.gethostname()


def handle_driver(conn, addr):
    """Send tasks to the driver and receive results"""
    print(f"[+] Driver connected: {addr}")

    try:
        while True:
            global pending_tasks

            try:
                task = task_queue.get(timeout=2)
            except Empty:
                break

            send_message(conn, task)
            print(f"Task {task.id} sent to {addr}")

            result = recv_message(conn)

            with results_lock:
                results[result.id] = result
                print(f"[RESULT] Task {result.id} from {addr}: E={result.energy:.6f}")

            with pending_lock:
                pending_tasks -= 1
                if pending_tasks == 0:
                    all_tasks_done.set()

    except Exception as e:
        print(f"[ERROR] {addr}: {e}")
    finally:
        conn.close()
        print(f"[-] Driver disconnected: {addr}")


def accept_drivers(server_socket):
    while not all_tasks_done.is_set():
        try:
            server_socket.settimeout(1)
            conn, addr = server_socket.accept()
            t = threading.Thread(target=handle_driver, args=(conn, addr), daemon=True)
            t.start()
        except TimeoutError:
            continue


def main():
    global pending_tasks

    pending_tasks = num_tasks = 10

    for i in range(num_tasks):
        task_queue.put(Task(id=i, coords=np.zeros((2, 3))))

    print(f"Created {num_tasks} tasks")

    bind_host = get_bind_host()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))

        actual_port = s.getsockname()[1]

        with open("server_info.txt", "w") as f:
            f.write(f"{bind_host}\n{actual_port}\n")
        print(f"Server info saved to server_info.txt: {bind_host}:{actual_port}")

        s.listen()
        print(f"Waiting for {num_tasks} tasks to complete...\n")

        # Launch an extra thread to accept driver connections
        accept_thread = threading.Thread(target=accept_drivers, args=(s,), daemon=True)
        accept_thread.start()

        all_tasks_done.wait()

        print(f"\n[✓] All {num_tasks} tasks completed!")
        print("Results summary:")
        for task_id in sorted(results.keys()):
            energy = results[task_id].energy
            print(f"  Task {task_id}: E={energy:.6f}")


if __name__ == "__main__":
    main()
