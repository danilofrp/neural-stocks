# <editor-fold
import time
import multiprocessing
from functools import partial

class T:
    def __init__(self):
        print self.__class__.__name__

class Test(T):
    t = None
    def __init__(self):
        T.__init__(self)

    def wait(self, t):
        self.t = t
        print self.t
        time.sleep(self.t)
        print self.t

def trainWrapper(n, obj):
    obj.wait(n)

def main():
    test = Test()

    def waitOut(n):
        return test.wait(n)

    num_processes = multiprocessing.cpu_count()
    a = range(1, num_processes+1)

    func = partial(trainWrapper, obj = test)
    p = multiprocessing.Pool(processes=num_processes)
    results = p.map(func, a)
    p.close()
    p.join()

if __name__ == "__main__":
    main()
# </editor-fold>
