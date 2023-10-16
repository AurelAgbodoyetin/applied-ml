from datetime import datetime
from multiprocessing import Pool
import time
from typing import Tuple

lst = [i for i in range(10_000)]


def mul(x: Tuple[int, int]):
    print(f"start process {x}")
    time.sleep(3)
    print(f"end process {x}")
    res = x[0] * x[1]
    res_ap = (x[0] , x[1] , res)
    return res_ap

def square(x: int):
    print(f"start process {x}")
    # time.sleep(3)
    print(f"end process {x}")
    res = x**x
    res_ap = (x , res)
    return res_ap

# Map
def test_map():
    pool = Pool(processes=10)
    result = pool.map(square, lst)
    # result = pool.map(mul, lst,chunksize=2)
    pool.close()
    pool.join()
    # print(result)

# For
def test_for():
    result = [i**i for i in lst]
    # print(result)


if __name__ == '__main__':
    start = datetime.now()
    test_map()
    print("End Time Map:", (datetime.now() - start).total_seconds())

    start = datetime.now()
    test_for()
    print("End Time For:", (datetime.now() - start).total_seconds())