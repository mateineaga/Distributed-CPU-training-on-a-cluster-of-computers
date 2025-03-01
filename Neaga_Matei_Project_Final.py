import os
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

# MPI config
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Hyperparameters
learning_rate = 0.001
epochs = 10
n_samples = 100
n_features = 10

class LinearModel:
    def __init__(self, n_features):
        self.W = np.random.rand(n_features, 1)
        self.b = np.random.rand(1)

    def forward(self, X):
        return X @ self.W + self.b

    def backward(self, X, y, preds):
        error = preds - y
        loss = np.mean(error ** 2)
        grad_W = 2 * X.T @ error / len(y)
        grad_b = 2 * np.mean(error)
        return loss, grad_W, grad_b

    def update(self, grad_W, grad_b, lr):
        self.W -= lr * grad_W
        self.b -= lr * grad_b

def generate_data(rank, size, n_samples, n_features):
    np.random.seed(rank)  # Seed specific pentru fiecare proces
    total_samples = n_samples
    local_samples = total_samples // size
    X = np.random.rand(local_samples, n_features)
    y = X @ np.random.rand(n_features, 1) + 0.1 * np.random.randn(local_samples, 1)
    return X, y

def train_distributed(rank, size, n_samples, n_features, epochs, lr):
    # Debugging info: numărul total de procese și rank-ul fiecărui proces
    if rank == 0:
        print(f"Starting distributed training with {size} processes.")
    print(f"Process {rank} is running on this node.")

    # Generare date locale pentru fiecare proces
    X, y = generate_data(rank, size, n_samples, n_features)

    model = LinearModel(n_features)
    loss_history = []
    sync_times = []
    contrib_W_history = []
    contrib_b_history = []

    for epoch in range(epochs):
        preds = model.forward(X)
        loss, grad_W, grad_b = model.backward(X, y, preds)

        # Reducere Allreduce pentru gradienti
        total_grad_W = np.zeros_like(grad_W)
        total_grad_b = np.zeros_like(grad_b)

        # Măsurare timp de sincronizare
        start_time = MPI.Wtime()
        comm.Allreduce(grad_W, total_grad_W, op=MPI.SUM)
        comm.Allreduce(grad_b, total_grad_b, op=MPI.SUM)
        end_time = MPI.Wtime()

        sync_time = end_time - start_time
        sync_times.append(sync_time)

        # Calcul contribuția fiecărui proces
        local_grad_norm_W = np.linalg.norm(grad_W)
        local_grad_norm_b = np.linalg.norm(grad_b)
        global_grad_norm_W = np.linalg.norm(total_grad_W)
        global_grad_norm_b = np.linalg.norm(total_grad_b)

        contrib_W = (local_grad_norm_W / global_grad_norm_W) * 100 if global_grad_norm_W > 0 else 0
        contrib_b = (local_grad_norm_b / global_grad_norm_b) * 100 if global_grad_norm_b > 0 else 0

        contrib_W_history.append(contrib_W)
        contrib_b_history.append(contrib_b)

        # Debugging info: gradientii calculați de fiecare proces, contribuțiile și timpul de sincronizare
        print(f"Process {rank}, Epoch {epoch+1}: grad_W mean = {np.mean(grad_W):.4f}, grad_b mean = {np.mean(grad_b):.4f}")
        print(f"Process {rank}, Epoch {epoch+1}: Contribution to W = {contrib_W:.2f}%, b = {contrib_b:.2f}%")
        print(f"Process {rank}, Epoch {epoch+1}: Sync time = {sync_time:.6f} seconds")

        # Actualizare model cu gradientii medii
        model.update(total_grad_W / size, total_grad_b / size, lr)

        # Sincronizare între procese
        comm.Barrier()

        # Rank 0 salvează și afișează loss-ul
        if rank == 0:
            loss_history.append(loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    # Salvare grafic contribuții pentru fiecare proces
    plt.figure()
    plt.plot(range(1, epochs + 1), contrib_W_history, label="Contribution to W")
    plt.plot(range(1, epochs + 1), contrib_b_history, label="Contribution to b")
    plt.xlabel("Epoch")
    plt.ylabel("Contribution (%)")
    plt.title(f"Contribution per Process (Rank {rank})")
    plt.legend()
    plt.savefig(f"contribution_plot_rank_{rank}.png")
    print(f"Saved contribution plot for rank {rank} as 'contribution_plot_rank_{rank}.png'")

    # Rank 0 salvează graficul și parametrii finali
    if rank == 0:
        # Salvare grafic pierdere
        plt.figure()
        plt.plot(range(1, epochs + 1), loss_history, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Distributed Training Loss")
        plt.legend()
        plt.savefig("loss_plot.png")
        print("Saved loss plot as 'loss_plot.png'")

        # Salvare grafic timp de sincronizare
        plt.figure()
        plt.plot(range(1, epochs + 1), sync_times, label="Sync Time")
        plt.xlabel("Epoch")
        plt.ylabel("Sync Time (s)")
        plt.title("Synchronization Time per Epoch")
        plt.legend()
        plt.savefig("sync_time_plot.png")
        print("Saved sync time plot as 'sync_time_plot.png'")

        # Salvare parametri finali în format .txt
        with open("model_parameters.txt", "w") as f:
            f.write("W:\n")
            np.savetxt(f, model.W)
            f.write("\nb:\n")
            np.savetxt(f, model.b)
        print("Saved model parameters to 'model_parameters.txt'")

        # Salvare timp de sincronizare
        with open("sync_times.txt", "w") as f:
            f.write("Epoch, Sync Time (s)\n")
            for epoch, time in enumerate(sync_times, 1):
                f.write(f"{epoch}, {time:.6f}\n")
        print("Saved synchronization times to 'sync_times.txt'")

        # Salvare contribuții
        with open("contributions.txt", "a") as f:
            for epoch in range(epochs):
                f.write(f"Epoch {epoch+1}, Process {rank}, W Contribution = {contrib_W:.2f}%, b Contribution = {contrib_b:.2f}%\n")

if __name__ == "__main__":
    train_distributed(rank, size, n_samples, n_features, epochs, learning_rate)
