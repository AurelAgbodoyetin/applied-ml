from multiprocessing import Pool
import time
from typing import Tuple
from timeit import timeit

lst = [i for i in range(10_000)]


def mul(x: Tuple[int, int]):
    # print(f"start process {x}")
    time.sleep(3)
    # print(f"end process {x}")
    res = x[0] * x[1]
    res_ap = (x[0] , x[1] , res)
    return res_ap

def square(x: int):
    # print(f"start process {x}")
    # time.sleep(3)
    # print(f"end process {x}")
    res = x**x
    res_ap = (x , res)
    return res_ap

# Map
@timeit
def test_map():
    pool = Pool(processes=10)
    result = pool.map(square, lst)
    # result = pool.map(mul, lst,chunksize=2)
    pool.close()
# For
@timeit
def test_for():
    result = [i**i for i in lst]
    # print(result)


if __name__ == '__main__':
    test_map()
    test_for()