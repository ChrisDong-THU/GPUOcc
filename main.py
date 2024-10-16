import os
import torch
import time
import argparse


def main(local_rank, args, interval=10):
    # 检查是否有可用的CUDA设备
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    
    torch.cuda.set_device(local_rank)
    device_properties = torch.cuda.get_device_properties(local_rank)
    
    # 获取总显存
    total_memory = device_properties.total_memory
    target_memory = total_memory * args.util
    reserved_memory = 500 * 1024 ** 2  # 预留 500MB 防止占满显存

    max_memory_to_allocate = target_memory - reserved_memory

    if local_rank == 0:
        print(f"Using device: {args.device}")
        print(f"Total GPU memory: {total_memory / (1024 ** 3):.2f} GB")
        print(f"Target memory utilization: {args.util * 100}%")
        print(f"Target memory to allocate: {max_memory_to_allocate / (1024 ** 3):.2f} GB")

    tensors = []

    # 尝试分配尽可能多的显存
    try:
        block_size = 1024 ** 2 * 100  # 每次分配100MB
        total_allocated = 0

        # 持续分配显存直到达到目标占用率
        while total_allocated < max_memory_to_allocate:
            tensors.append(torch.empty(block_size // 4, dtype=torch.float32).cuda())
            total_allocated += block_size

    except RuntimeError as e:
        print(f"Memory allocation stopped due to: {e}")
    finally:
        print(f"Total allocated memory of {local_rank}: {total_allocated / (1024 ** 3):.2f} GB")
    
    # 持续运行，保持内存占用
    print(f"Maintain utilization of {local_rank} ...")
    while True:
        # 对张量进行简单的加法操作，保持GPU忙碌，避免长时间闲置导致释放显存
        for tensor in tensors:
            tensor.add_(1)

        # 休眠指定时间，然后继续操作
        time.sleep(interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPU occupancy stress test')
    parser.add_argument('--device', type=str, default="6,7", help='CUDA device to use')
    parser.add_argument('--util', type=float, default=0.6, help='Target GPU memory utilization')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    ngpus = torch.cuda.device_count()

    try:
        if ngpus > 1:
            torch.multiprocessing.spawn(main, args=(args,), nprocs=ngpus)
        else:
            main(0, args)
    except KeyboardInterrupt:
        print("Stopped by user.")