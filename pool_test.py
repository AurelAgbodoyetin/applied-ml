import concurrent
import time
from multiprocessing import shared_memory
import numpy as np
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor

from utils import divide_chunks

NUM_WORKERS = multiprocessing.cpu_count()
np.random.seed(42)
ARRAY_SIZE = int(15)
ARRAY_SHAPE = (ARRAY_SIZE,)
NP_SHARED_NAME = 'npshared'
NP_DATA_TYPE = np.float64
data = np.arange(ARRAY_SIZE, dtype=NP_DATA_TYPE).reshape(ARRAY_SHAPE)


def create_shared_memory_nparray(data):
    d_size = np.dtype(NP_DATA_TYPE).itemsize * np.prod(ARRAY_SHAPE)
    shm = shared_memory.SharedMemory(create=True, size=d_size, name=NP_SHARED_NAME)
    # numpy array on shared memory buffer
    dst = np.ndarray(shape=ARRAY_SHAPE, dtype=NP_DATA_TYPE, buffer=shm.buf)
    dst[:] = data[:]
    print(f'NP SIZE: {(dst.nbytes / 1024) / 1024}')
    return shm, dst


def release_shared(name):
    shm = shared_memory.SharedMemory(name=name)
    dst = np.ndarray(shape=ARRAY_SHAPE, dtype=NP_DATA_TYPE, buffer=shm.buf, create=False).copy()
    shm.close()
    shm.unlink() 
    return dst


def np_square(indices):
    # not mandatory to init it here, just for demostration purposes.
    shm = shared_memory.SharedMemory(name=NP_SHARED_NAME)
    np_array = np.ndarray(ARRAY_SHAPE, dtype=NP_DATA_TYPE, buffer=shm.buf)
    #print(locals())
    for i in indices:
        np_array[i] = np_array[i]**2
        #print(np_array[i])

  
def benchmark():
    chunks = list(divide_chunks(list(range(len(data))), NUM_WORKERS))
    start_time = time.time_ns()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(np_square, chunks)

    print((time.time_ns() - start_time) / 1_000_000)

  
if __name__ == '__main__':
    print(data)
    shm, sharr = create_shared_memory_nparray(data)
    benchmark()
    res = release_shared(NP_SHARED_NAME)
    print(res)