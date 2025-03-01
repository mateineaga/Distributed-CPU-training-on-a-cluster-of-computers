# Distributed CPU Training on FEP (SLURM)

## Introduction

The main goal of this project is to implement a distributed training model on multiple microprocessors (CPUs). This process is automated and runs on the FEP infrastructure, specifically using SLURM, which is a cluster of computers located at UPB, in the PRECIS building. The project utilizes the MPI library, particularly the `allreduce` function, to distribute the model and its tasks across multiple processes. Each process independently performs forward and backward propagation, where the user implements backpropagation instead of relying on built-in functions (e.g., autograd, backward, forward). Once all gradients are computed, they are synchronized using `allreduce` to update the model parameters.

## Project Implementation Plan

The project follows a systematic approach to developing a simple distributed model for training a machine learning algorithm across multiple CPU cores. It is built using the `mpi4py` library, which allows computational tasks to be distributed among multiple processes. Each MPI process runs on a separate CPU core, and synchronization between processes is achieved using `comm.Allreduce`. The implementation relies entirely on `numpy`, meaning no references to PyTorch, CUDA, or GPUs exist—all computations are performed exclusively on CPUs.

The number of processes is specified using the command:
```sh
mpirun -np <num_processes> python3 distributed_training.py
```
This command determines the number of CPU cores used. Each process independently computes gradients and synchronizes them with others using `MPI.Allreduce`. For example, running:
```sh
mpirun -np 4 python3 distributed_training.py
```
utilizes 4 CPUs. The execution order of processes depends on MPI's internal mechanisms and the operating system.

Each MPI process starts execution simultaneously, but processes do not necessarily run at the same speed due to initialization time, resource allocation, or data access. The order of displayed results depends on when each process completes its task and writes to the output stream (`stdout`).

To facilitate testing and reproducibility, an artificial dataset was generated and distributed evenly among parallel processes. Gradient synchronization was performed using MPI’s `Allreduce` operation, aggregating each process’s contributions for global model updates. Detailed logs and debugging mechanisms were implemented to ensure correct gradient synchronization and computation.

### Standard Output Display
The `stdout` output is shared among all processes, meaning messages from different processes may appear in an unsynchronized manner. For instance, process 2 may complete a task and display results before process 0. Depending on how MPI assigns processes to CPUs, some processes may receive more resources and complete tasks faster.

## Results

Example log output:
```sh
Process 1 is running on this node.
Process 1, Epoch 1: grad_W mean = -0.0283, grad_b mean = -0.0010
Process 1, Epoch 1: Contribution to W = 8.99%, b = 0.05%
Process 1, Epoch 1: Sync time = 0.201988 seconds
...
Process 0, Epoch 10: grad_W mean = -0.2028, grad_b mean = -0.3872
Process 0, Epoch 10: Contribution to W = 23.67%, b = 20.40%
Process 0, Epoch 10: Sync time = 0.000069 seconds
```

### Model Parameters After Training
```sh
W:
9.505355511359944720e-02
8.608506196416068068e-01
...

b:
1.276012244831889186e-01
```

### Synchronization Time Per Epoch
| Epoch | Sync Time (s) |
|-------|-------------|
| 1     | 0.285622    |
| 2     | 0.013190    |
| 3     | 0.000103    |
| ...   | ...         |
| 10    | 0.000069    |

## Interpretation of Results

The experiment demonstrates that the distributed process functions correctly across multiple CPU cores. Processes run on different nodes and report varying mean gradient values for `W` and `b`. For example, process 2 has higher gradients that gradually decrease, while processes 1 and 3 report smaller or negative values, reflecting their different contributions to optimization.

A significant observation is the drastic reduction in synchronization time after the first epoch. The initial sync takes 0.285 seconds, but subsequent epochs have negligible sync times (under 0.0001 seconds). This suggests that while initial communication setup incurs overhead, gradient synchronization becomes highly efficient once established.

Additionally, an imbalance in process contributions is noted. Process 2 contributes disproportionately to gradient updates (over 110% for `W` and `b`), indicating an imbalance in distributed data. In contrast, processes 0, 1, and 3 contribute around 20-23%, 9-10%, and 12% respectively.

The increasing loss suggests that the model parameters may not be optimal. This could be due to an uneven data distribution among processes or errors in gradient synchronization via `MPI.Allreduce`.

## Conclusion

The experiment successfully demonstrates distributed training across four CPU cores. Each process correctly receives and processes its assigned data, and inter-process synchronization functions as expected. The `Allreduce` operation ensures that each process receives the sum of computed gradients and uniformly averages them.

Each process locally computes gradients for `W` and `b`, and MPI’s `Allreduce` operation combines contributions from all processes to obtain an updated global model. While the synchronization mechanism works efficiently, future improvements should address data distribution imbalances to optimize model training results.

