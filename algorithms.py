import numpy as np
from tqdm import tqdm

def generate_sparse_vector(N_samples, m, sparsity_level):
    Z = np.random.randn(m, N_samples)
    zero_indices = np.array([np.random.choice(m, int(sparsity_level * m), replace=False) for _ in range(N_samples)])
    Z[zero_indices.T, np.arange(N_samples)] = 0
    return Z

def L2_loss(Wd, Z, X):
    return (0.5 * np.square(X - Wd @ Z).mean(axis=0)).mean()


def sparse_code_loss(Wd, Z, X, alpha):
    return (0.5 * np.square(X - Wd @ Z).mean(axis=0) + alpha * np.abs(Z).sum(axis=0)).mean()


def MSE(Z_true, Z_pred):
    # Mean Squared Error between the true target Z_true and predicted Z_pred
    return np.mean(np.square(Z_true - Z_pred))


def get_max_eigenvalue(WdTWd):
    eigenvalues, eigenvectors = np.linalg.eig(WdTWd)
    eigenvalues = np.real(eigenvalues)
    return eigenvalues.max()


def subgradient_method(Wd, X, alpha, m, N_samples, n_iter=1000, start='random'):
    if start == 'random': Z = np.random.randn(m, N_samples) # (m,N_samples)
    elif start == 'zeros': Z = np.zeros((m, N_samples)) # (m,N_samples)
    else: return None
    WdTWd = Wd.T @ Wd
    L = get_max_eigenvalue(WdTWd) + 1
    We = Wd.T * 1 / L # (m,n)
    S = np.identity(m) - WdTWd * 1 / L # (m,m)
    Z_arr = [Z]
    for _ in tqdm(range(n_iter)):
        Z = S @ Z + We @ X - alpha / L * np.sign(Z)
        Z_arr.append(Z)
    return Z_arr


def ISTA(Wd, X, alpha, m, N_samples, n_iter=1000, start='random'):
    # economic implementation
    if start == 'random': Z = np.random.randn(m, N_samples)  # (m,N_samples)
    elif start == 'zeros': Z = np.zeros((m, N_samples))  # (m,N_samples)
    else: return None
    WdTWd = Wd.T @ Wd
    L = get_max_eigenvalue(WdTWd) +1
    We = Wd.T * 1 / L  # (m,n)
    S = np.identity(m) - WdTWd * 1 / L  # (m,m)
    shrinkage = lambda v: np.sign(v) * np.max([np.abs(v) - alpha/L, np.zeros(v.shape)], axis=0)
    Z_arr = [Z]
    for _ in tqdm(range(n_iter)):
        Z = shrinkage(S @ Z + We @ X)
        Z_arr.append(Z)
    return Z_arr


def ISTA_slow(Wd, X, alpha, m, N_samples, n_iter=1000, start='random'):
    # straightforward implementation
    if start == 'random': Z = np.random.randn(m, N_samples) # (m,N_samples)
    elif start == 'zeros': Z = np.zeros((m, N_samples)) # (m,N_samples)
    else: return None
    WdTWd = Wd.T @ Wd
    L = get_max_eigenvalue(WdTWd) + 1
    shrinkage = lambda v: np.sign(v) * np.max([np.abs(v) - alpha/L, np.zeros(v.shape)], axis=0)
    Z_arr = [Z]
    for _ in tqdm(range(n_iter)):
        Z = shrinkage(Z - 1/L * (Wd.T @ (Wd @ Z - X)))
        Z_arr.append(Z)
    return Z_arr


def FISTA(Wd, X, alpha, m, N_samples, n_iter=1000, start='random'):
    # economic implementation
    if start == 'random': Z = np.random.randn(m, N_samples) # (m,N_samples)
    elif start == 'zeros': Z = np.zeros((m, N_samples)) # (m,N_samples)
    else: return None
    WdTWd = Wd.T @ Wd
    L = get_max_eigenvalue(WdTWd) +1
    We = Wd.T * 1 / L # (m,n)
    S = np.identity(m) - WdTWd * 1 / L # (m,m)
    shrinkage = lambda v: np.sign(v) * np.max([np.abs(v) - alpha/L, np.zeros(v.shape)], axis=0)
    Z_arr = [Z]

    new_t = lambda t: 0.5 * (1 + np.sqrt(1 + 4 * (t**2)))
    t = 0
    Z1 = shrinkage(S @ Z + We @ X)
    Z1_arr = [Z1]

    for _ in tqdm(range(n_iter)):
        Z1_arr.append(shrinkage(S @ Z + We @ X))
        Z = Z1_arr[-1] + (t - 1) / new_t(t) * (Z1_arr[-1] - Z1_arr[-2])
        t = new_t(t)
        Z_arr.append(Z)
    return Z_arr


def CoD(Wd, X, alpha, m, N_samples, n_iter=1000, start='random', print_loss=True):
    if start == 'random': Z = np.random.randn(m, N_samples) # (m,N_samples)
    elif start == 'zeros': Z = np.zeros((m, N_samples)) # (m,N_samples)
    else: return None
    WdTWd = Wd.T @ Wd
    L = get_max_eigenvalue(WdTWd) + 1
    We = Wd.T * 1 / L # (m,n)
    S = np.identity(m) - WdTWd * 1 / L # (m,m)
    B = We @ X  # (m,N_samples)
    shrinkage = lambda v: np.sign(v) * np.max([np.abs(v) - alpha / L, np.zeros(v.shape)], axis=0)
    Z_arr = [Z]
    if print_loss: pbar = tqdm(range(n_iter))
    else: pbar = range(n_iter)
    for _ in pbar:
        Z = Z.copy()
        Z_overline = shrinkage(B)  # (m,N_samples)
        diff = Z_overline - Z  # (m,N_samples)
        # for i in range(N_samples):
        #     for j in range(m):
        #         k = np.argmax(np.abs(diff[:, i]))
        #         # for k in range(m):
        #         B[j, i] = B[j, i] + S[k, j] * diff[k, i]
        #         Z[k] = Z_overline[k]
        k = np.argmax(np.abs(diff), axis=0) # (N_samples)
        B = B + S[k].T * diff[k, np.arange(N_samples)] # (m,N_samples)
        Z[k] = Z_overline[k] # (m,N_samples)
        Z_arr.append(Z)

    Z = shrinkage(B)
    Z_arr.append(Z)
    return Z_arr


#############################################
######### DL algorithms #####################
#############################################
import torch
import torch.nn as nn
import torch.optim as optim
# Loss Function
def MSE_tensor(Z_true, Z_pred):
    # Mean Squared Error between the true target Z_true and predicted Z_pred
    return torch.mean(torch.square(Z_true - Z_pred))


def train_encoder(model, X_train, Z_true_train, X_val, Z_true_val, n_iter=1000, learning_rate=0.001, scheduler=None):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Z_true_train = torch.tensor(Z_true_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Z_true_val = torch.tensor(Z_true_val, dtype=torch.float32)

    # Define the optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if scheduler == 'linear':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda t: learning_rate/(t+1))
        lr = scheduler.get_last_lr()[0]
    elif scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        lr = scheduler.get_last_lr()[0]
    else: lr = learning_rate

    # Training loop
    losses = {'train': [], 'val': [], 'W': []}

    # Validation phase
    def validate(model):
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No need to track gradients for validation
            Z_pred_val = model(X_val)
            val_loss = MSE_tensor(Z_true_val, Z_pred_val)
            losses['val'].append(val_loss.item())
        return val_loss
    val_loss = validate(model)

    pbar = tqdm(range(n_iter))
    def pbar_update(epoch, loss, val_loss, lr):
        pbar.set_description(f"Epoch [{epoch+1}/{n_iter}], Train loss: {loss.item():.4f}, Val loss: {val_loss.item():.4f}, LR: {lr:.4f}")

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        # Forward pass: Get predicted Z
        Z_pred = model(X_train)

        # Compute loss
        loss = MSE_tensor(Z_true_train, Z_pred)
        losses['train'].append(loss.item())
        losses['W'].append(model.W.weight.detach().numpy().copy())

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Validation phase
        val_loss = validate(model)

        if epoch % 10 == 0: pbar_update(epoch, loss, val_loss, lr)

        if scheduler is not None:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

    pbar_update(epoch, loss, val_loss, lr)
    return model, losses


# DL baseline encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Trainable W matrix (n x m)
        self.W = nn.Parameter(torch.randn(input_dim, output_dim, dtype=torch.float32))
        # Trainable diagonal matrix D (m x m)
        self.D = nn.Parameter(torch.randn(output_dim, output_dim, dtype=torch.float32))

    def forward(self, X):
        # Forward pass: Z = D * tanh(W*X)
        WX = torch.matmul(self.W.T, X)  # W*X  (m,N_samples)
        Z = torch.matmul(self.D, torch.tanh(WX))  # Apply g_D(y) = D * tanh(y)  #
        return Z


class LISTA(nn.Module):
    def __init__(self, input_dim, output_dim, L=3, W=None, alpha=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.L = L
        # Trainable W matrix (n x m)
        self.W = nn.Linear(input_dim, output_dim, dtype=torch.float32)
        # Trainable S matrix (m x m)
        self.S = nn.Linear(output_dim, output_dim, dtype=torch.float32)
        # Trainable theta vector (m, 1)
        scale = 1.001 * np.linalg.norm(W, ord=2)**2
        self.theta = nn.Parameter(torch.tensor([alpha / scale] * self.output_dim, dtype=torch.float32))
        self.norm_layer = nn.BatchNorm1d(self.output_dim)

    def shrinkage(self, v):
        return torch.sign(v) * torch.clamp(torch.abs(v) - self.theta, min=0)

    def forward(self, X):
        # Forward pass: Z_t+1 = h_theta(S * Z_t + W * X)) for t=1,...,T, Z0 = 0
        B = self.W(X.T)
        Z = torch.zeros(X.shape[1], self.output_dim, dtype=torch.float32)
        for t in range(self.L):
            Z = self.norm_layer(self.shrinkage(self.S(Z) + B))
        return Z.T


class LISTA_diff_params(nn.Module):
    def __init__(self, input_dim, output_dim, L=1, W=None, alpha=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.L = L
        # Trainable W matrix (n x m)
        self.W = nn.Linear(input_dim, output_dim, dtype=torch.float32)
        # Trainable S matrix (m x m x L)
        self.Ss = [nn.Linear(output_dim, output_dim, L, dtype=torch.float32) for l in range(L)]
        # Trainable theta vector (m x 1 x L)
        scale = 1.001 * np.linalg.norm(W, ord=2)**2
        self.thetas = [nn.Parameter(torch.tensor([alpha / scale] * self.output_dim, dtype=torch.float32)) for l in range(L)]
        self.norm_layer = nn.BatchNorm1d(self.output_dim)

    def shrinkage(self, v, theta):
        return torch.sign(v) * torch.max(torch.abs(v) - theta, torch.zeros_like(v))

    def forward(self, X):
        # Forward pass: Z_t+1 = h_theta(S * Z_t + W * X)) for t=1,...,T, Z0 = 0
        B = self.W(X.T)
        Z = torch.zeros(X.shape[1], self.output_dim, dtype=torch.float32)
        for t in range(self.L):
            Z = self.norm_layer(self.shrinkage(self.Ss[t](Z) + B, self.thetas[t]))
        return Z.T


#############################################
################ Online #####################
#############################################


#%%
def learn_dictionary_Wd(data_generator, alpha, n, m, T=100, batch_size=1, n_iter=1000, start='zeros'):
    Wd = np.random.randn(n, m)
    Wd = Wd / np.linalg.norm(Wd, axis=0)
    diffs = []
    for t in range(1,T+1):
        # (1) get an image patch from the training set Xp;
        train_samples = data_generator(batch_size)
        # (2) calculate the optimal code Z∗p using the CoD Algorithm 2;
        sparse_code = CoD(Wd, train_samples, alpha, m, batch_size, n_iter=n_iter, start=start)[-1]
        # (3) update Wd with one step of stochastic gradient
        old_Wd = Wd
        eta = 1 / t
        Wd = Wd + eta * (train_samples - Wd @ sparse_code) @ sparse_code.T / batch_size
        # (4) renormalize the column of Wd to unit norm;
        Wd = Wd / np.linalg.norm(Wd, axis=0)
        diffs.append(np.square(Wd - old_Wd).mean())
    return Wd, diffs


def train_encoder_online(data_generator, model, Wd, alpha, n, m, T=100, learning_rate=0.001, N_val=50,
                         batch_size=1, n_iter=1000, start='zeros'):
    # prepare validation set
    val_set = data_generator(N_val)
    val_sparse_code = CoD(Wd, val_set, alpha, m, N_val, n_iter=1000, start='zeros')[-1]
    val_set = torch.tensor(val_set, dtype=torch.float32)
    val_sparse_code = torch.tensor(val_sparse_code, dtype=torch.float32)

    # Define the optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda t: learning_rate/(t+1))

    # Training loop
    losses = {'train': [], 'val': [], 'W': []}

    # Validation phase
    def validate(model):
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No need to track gradients for validation
            Z_pred_val = model(val_set)
            val_loss = MSE_tensor(val_sparse_code, Z_pred_val)
            losses['val'].append(val_loss.item())
        return val_loss
    val_loss = validate(model)

    pbar = tqdm(range(T))
    def pbar_update(epoch, loss, val_loss, lr):
        pbar.set_description(f"Epoch [{epoch+1}/{n_iter}], Train loss: {loss.item():.4f}, Val loss: {val_loss.item():.4f}, LR: {lr:.4f}")


    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        # (1) get an image patch from the training set Xp;
        train_samples = data_generator(batch_size)
        # (2) calculate the optimal code Z∗p using the CoD Algorithm 2;
        sparse_code = CoD(Wd, train_samples, alpha, m, batch_size, n_iter=n_iter, start=start, print_loss=False)[-1]
        # (3) Forward pass: Get predicted Z
        Z_pred = model(torch.tensor(train_samples, dtype=torch.float32))
        # Compute loss
        loss = MSE_tensor(torch.tensor(sparse_code, dtype=torch.float32), Z_pred)
        losses['train'].append(loss.item())
        losses['W'].append(model.W.weight.detach().numpy().copy())
        # (4,5) perform bprop update the encoder parameters;
        loss.backward() # Backpropagation
        optimizer.step() # Update weights

        # Validation loss
        val_loss = validate(model)

        if epoch % 10 == 0: pbar_update(epoch, loss, val_loss, scheduler.get_last_lr()[0])

        # scheduler.step()

    pbar_update(epoch, loss, val_loss, scheduler.get_last_lr()[0])
    return model, losses