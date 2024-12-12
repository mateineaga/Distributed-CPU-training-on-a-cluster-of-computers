import os
import numpy as np
from multiprocessing import Process
from mpi4py import MPI

# MPI config
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  
size = comm.Get_size()  

# Hyperparameters
learning_rate = 0.01
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
    np.random.seed(rank)  
    total_samples = n_samples
    local_samples = total_samples // size
    X = np.random.rand(local_samples, n_features)
    y = X @ np.random.rand(n_features, 1) + 0.1 * np.random.randn(local_samples, 1)
    return X, y


def train_process(rank, size, n_samples, n_features, epochs, lr):

    X, y = generate_data(rank, size, n_samples, n_features)

    model = LinearModel(n_features)

    for epoch in range(epochs):
        preds = model.forward(X)

        loss, grad_W, grad_b = model.backward(X, y, preds)


        total_grad_W = np.zeros_like(grad_W)
        total_grad_b = np.zeros_like(grad_b)
        comm.Allreduce(grad_W, total_grad_W, op=MPI.SUM)
        comm.Allreduce(grad_b, total_grad_b, op=MPI.SUM)

        model.update(total_grad_W / size, total_grad_b / size, lr)


        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    if rank == 0:
        print(f"Training finished. Final parameters:\nW: {model.W}\nb: {model.b}")

if __name__ == "__main__":
    processes = []

    for i in range(size):

        pid = os.fork()
        if pid == 0:

            train_process(rank=i, size=size, n_samples=n_samples, n_features=n_features, epochs=epochs, lr=learning_rate)
            os._exit(0)  

    for _ in range(size):
        os.wait()
