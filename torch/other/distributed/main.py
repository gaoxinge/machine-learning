import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def simple(rank, size, lock):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        req = dist.isend(tensor=tensor, dst=1)
        with lock:
            print('rank 0 started sending')
    else:
        req = dist.irecv(tensor=tensor, src=0)
        with lock:
            print('rank 1 started receiving')
    req.wait()
    with lock:
        print(f'rank {rank} has data {tensor}')


def all_reduce(rank, size, lock):
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    with lock:
        print(f'rank {rank} has data {tensor}')


def init_process(rank, size, fn, lock, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '9999'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, lock)


if __name__ == "__main__":
    size = 2
    fn = all_reduce
    lock = mp.Lock()

    ps = []
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, fn, lock))
        ps.append(p)

    for p in ps:
        p.start()

    for p in ps:
        p.join()

