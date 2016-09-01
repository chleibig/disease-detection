"""This code is based on keras's solution to running a slow source data
   generator on CPU in parallel to network training on GPU"""
import multiprocessing
import Queue as queue
import time
import threading
import numpy as np


def generator_queue(generator, max_q_size=10,
                    wait_time=0.05, nb_worker=1, pickle_safe=False):
    """Builds a queue out of a data generator.
    If pickle_safe, use a multiprocessing approach. Else, use threading.
    """

    generator_threads = []
    if pickle_safe:
        q = multiprocessing.Queue(maxsize=max_q_size)
        _stop = multiprocessing.Event()
    else:
        q = queue.Queue()
        _stop = threading.Event()

    try:
        def data_generator_task():
            while not _stop.is_set():
                try:
                    if q.qsize() < max_q_size:
                        try:
                            generator_output = next(generator)
                        except ValueError:
                            continue
                        q.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except Exception:
                    _stop.set()
                    raise

        for i in range(nb_worker):
            if pickle_safe:
                # Reset random seed else all children processes share the same
                # seed
                np.random.seed()
                thread = multiprocessing.Process(target=data_generator_task)
            else:
                thread = threading.Thread(target=data_generator_task)
            generator_threads.append(thread)
            thread.daemon = True
            thread.start()
    except:
        _stop.set()
        if pickle_safe:
            # Terminate all daemon processes
            for p in generator_threads:
                if p.is_alive():
                    p.terminate()
            q.close()
        raise

    return q, _stop


