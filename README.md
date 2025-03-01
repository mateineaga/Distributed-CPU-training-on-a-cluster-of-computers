# 🚀 Distributed CPU Training on FEP (SLURM)

## 🏆 Introduction

The main goal of this project is to implement a **distributed training model** on multiple **microprocessors (CPUs)**. This process is automated and runs on the **FEP infrastructure**, specifically using **SLURM**, which is a **cluster of computers located at UPB, in the PRECIS building**. 

🛠️ **Technology Stack:**
- **MPI (`mpi4py`)** for parallel processing 🔄
- **SLURM Cluster** for distributed training ⚡
- **NumPy** for numerical computations 📊

Each process **independently performs forward and backward propagation**, where the user implements **backpropagation** instead of relying on built-in functions (e.g., `autograd`, `backward`, `forward`). Once all gradients are computed, they are synchronized using `allreduce` to update the model parameters.

---

## 📌 Project Implementation Plan

This project follows a **systematic approach** to developing a **distributed machine learning model** across **multiple CPU cores**. It is built using the `mpi4py` library, which allows computational tasks to be distributed among multiple processes. 

🔥 **Key Features:**
✔️ Each MPI process runs on a **separate CPU core** 🏗️  
✔️ Synchronization between processes is achieved using **`MPI.Allreduce`** 🔄  
✔️ No **PyTorch, CUDA, or GPUs** involved—all computations are done on **CPUs only** 🖥️

The number of processes is specified using the command:
```sh
mpirun -np <num_processes> python3 distributed_training.py
```
💡 **Example:** Running **4 processes**:
```sh
mpirun -np 4 python3 distributed_training.py
```
Each process computes gradients **independently** and synchronizes them using `MPI.Allreduce`.

### ⚡ Standard Output Display
🖥️ **Important Notes:**
- The `stdout` output is **shared** among all processes 📢
- Messages from **different processes may appear out of order** due to execution speeds ⏳
- Some processes may receive **more resources**, completing tasks faster ⚡

---

## 📊 Results

### 📜 Example Log Output:
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

### 📌 Model Parameters After Training:
```sh
W:
9.505355511359944720e-02
8.608506196416068068e-01
...

b:
1.276012244831889186e-01
```

### ⏱️ Synchronization Time Per Epoch
| 🏷️ Epoch | ⏳ Sync Time (s) |
|---------|----------------|
| 1️⃣  | 0.285622 |
| 2️⃣  | 0.013190 |
| 3️⃣  | 0.000103 |
| ...  | ...        |
| 🔟  | 0.000069 |

---

## 🔎 Interpretation of Results

📌 **Key Observations:**
✔️ The **distributed process functions correctly** across multiple **CPU cores** 🔄  
✔️ Some processes contribute **more** to gradient updates than others, indicating **data imbalance** ⚠️  
✔️ **Synchronization time drops significantly** after the first epoch ⏱️  
✔️ The model parameters may not be **optimal**, possibly due to **uneven data distribution** 📉  

📢 **Process Contributions:**
- Process **2 contributes disproportionately** (>110% for `W` and `b`), meaning **data balancing is needed** ⚖️
- Processes **0, 1, and 3** contribute **20-23%, 9-10%, and 12%** respectively 🔄

---

## 🏁 Conclusion

✅ **Project Achievements:**
✔️ Successfully demonstrated **distributed training** across multiple **CPU cores** 🏗️  
✔️ Verified that each process **correctly receives and processes its assigned data** 📊  
✔️ Implemented an efficient **MPI-based synchronization mechanism** 🔄  

🚀 **Future Improvements:**
- **Optimize data distribution** among processes for better **gradient contribution balance** 📉  
- **Investigate hyperparameter tuning** to improve **model convergence** 🔬  
- **Enhance debugging mechanisms** for tracking individual process contributions 🛠️  

---

## 🛡️ References
📌 For full references, please see the [Full PDF Document](https://github.com/mateineaga/Distributed-CPU-training-on-a-cluster-of-computers/blob/main/Distributed-CPU-training-on-a-cluster-of-computers.pdf).

---