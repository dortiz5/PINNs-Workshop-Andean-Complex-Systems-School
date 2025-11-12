# Import NumPy for numerical operations
import numpy as np
# Import PyTorch for building and training neural networks
import torch
import torch.nn as nn
import torch.optim as optim
# Import Matplotlib for plotting
import matplotlib.pyplot as plt
import matplotlib as mlp
# Import the time module to time our training process
import time
# Ignore Warning Messages
import warnings
warnings.filterwarnings("ignore")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Actualización de los parámetros de Matplotlib
gray = '#5c5c5c' #'#5c5c5c' '000'
mlp.rcParams.update(
    {
        "image.cmap" : 'viridis', # plasma, inferno, magma, cividis
        "text.color" : gray,
        "xtick.color" :gray,
        "ytick.color" :gray,
        "axes.labelcolor" : gray,
        "axes.edgecolor" :gray,
        "axes.spines.right" : False,
        "axes.spines.top" : False,
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,

        'font.size' : 15,
        'interactive': False,
        "font.family": 'sans-serif',
        "legend.loc" : 'best',
        'text.usetex': False,
        'mathtext.fontset': 'stix',
    }
)

from scipy.integrate import solve_ivp

#===============================================================================
# ARQUITECTURA MODIFICADA
#===============================================================================

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class NeuralNetwork(nn.Module):

    def __init__(self, hlayers, fourier_dim=None, sigma=1.0):
        """
        Args:
            hlayers (list): lista con el número de neuronas en cada capa.
            fourier_dim (int): dimensión de las Fourier features (opcional).
            sigma (float): escala de las frecuencias aleatorias.
        """
        super(NeuralNetwork, self).__init__()

        self.fourier_dim = fourier_dim
        self.sigma = sigma

        if self.fourier_dim is not None:
            # Inicializamos matriz B ~ N(0, sigma^2)
            input_dim = hlayers[0]
            B = torch.randn((fourier_dim, input_dim)) * sigma
            self.register_buffer("B", B)  # se guarda como parte del modelo pero no se entrena
            # actualizamos la entrada de la red: ahora 2*fourier_dim
            hlayers = [2 * fourier_dim] + hlayers[1:]

        layers = []
        for i in range(len(hlayers[:-2])):
            layers.append(nn.Linear(hlayers[i], hlayers[i+1]))
            layers.append(Sin())
        layers.append(nn.Linear(hlayers[-2], hlayers[-1]))

        self.layers = nn.Sequential(*layers)
        self.init_params()

    def fourier_features(self, x):
        """Mapeo de Fourier features"""
        # x shape: [batch, input_dim]
        x_proj = 2 * torch.pi * x @ self.B.T  # [batch, fourier_dim]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def init_params(self):
        """Xavier Glorot parameter initialization of the Neural Network"""
        def init_normal(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        self.apply(init_normal)

    def forward(self, x):
        if self.fourier_dim is not None:
            x = self.fourier_features(x)
        return self.layers(x)
    
#===============================================================================
# SOLUCIÓN NUMÉRICA
#===============================================================================

def numerical_sol_lorenz(t_eval, sigma = 10, rho = 28, beta = 8.0/3.0,
                         x0 = 1, y0 = 1, z0 = 1):
        
    def lorenz_system(t, state, sigma, rho, beta):
        x, y, z = state
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        return [dx_dt, dy_dt, dz_dt]

    # Resolver con método numérico
    t_span = (0, T)
    t_eval_np = t_eval.detach().cpu().numpy().ravel()
    initial_state = [x0, y0, z0]

    sol = solve_ivp(
        lorenz_system, 
        t_span, 
        initial_state,
        args=(sigma, rho, beta),
        t_eval=t_eval_np,
        method='RK45'
    )

    x_true = torch.tensor(sol.y[0],device=device, dtype=torch.float32).view(-1, 1)
    y_true = torch.tensor(sol.y[1],device=device, dtype=torch.float32).view(-1, 1)
    z_true = torch.tensor(sol.y[2],device=device, dtype=torch.float32).view(-1, 1)
    
    return x_true, y_true, z_true


def crear_dominio_temporal(T, N_train=101, N_eval=1000):
    """Crea el dominio temporal para la PINN."""
    t_train = torch.linspace(0, T, N_train, 
                             device=device, 
                             requires_grad=True).view(-1, 1)  # entrenamiento
    t_eval = torch.linspace(0, T, N_eval, 
                             device=device, 
                             requires_grad=True).view(-1, 1)  # evaluación
    return t_train, t_eval # dominio de evaluación


def grad(outputs, inputs):
    """Computes the partial derivative of an output with respect
    to an input.
    Args:
        outputs: (N, 1) tensor
        inputs: (N, D) tensor
    """
    return torch.autograd.grad(outputs, inputs,
                        grad_outputs=torch.ones_like(outputs),
                        create_graph=True,
                        )[0]

def pinn_opimizer(pinn, lr = 0.01):

    # Define an optimizer (Adam) for training the network
    return optim.Adam(pinn.parameters(), lr=lr,
                        betas= (0.99,0.999), eps = 1e-8)


#===============================================================================
# ETAPA 1: DEFINICIÓN DE LOS PARÁMETROS (MODELO FÍSICO)
#===============================================================================

sigma = 10.0
rho = 28.0
beta = 8.0/3.0

x0 = 1.0
y0 = 1.0
z0 = 1.0

# Dominio temporal
T = 20.0        # tiempo total de simulación
N_train = 200   # puntos de colocación para entrenamiento
N_eval = 2000   # puntos para evaluación

#===============================================================================
# ETAPA 2: DEFINICIÓN DEL DOMINIO 
#===============================================================================
t_train, t_eval = crear_dominio_temporal(T, N_train, N_eval)

x_train, y_train, z_train = numerical_sol_lorenz(t_train)


#===============================================================================
# ETAPA 3: CREACIÓN DE LA RED NEURONAL SURROGANTE 
#===============================================================================

# Crear la red
torch.manual_seed(123)
hidden_layers = [1, 100, 100, 100, 3]
lorenz_pinn = NeuralNetwork(hidden_layers).to(device)
nparams = sum(p.numel() for p in lorenz_pinn.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {nparams}')

#==========================================================================
# ETAPA 4 Y 5: DEFINICIÓN DE LA FUNCIÓN DE COSTO BASADA EN AUTOGRAD
#==========================================================================
MSE = nn.MSELoss()

def LorenzPINNLoss(PINN, t_phys, sigma, rho, beta, 
                   x0=1.0, y0=1.0, z0=1.0,
                   lambda_pde=10.0, lambda_ic=10.0):
    
    t0 = torch.tensor(0., device=device, requires_grad=True).view(-1, 1)
    
    # Predicciones de la red
    u = PINN(t_phys)  # (N, 3)
    x, y, z = u[:, 0:1], u[:, 1:2], u[:, 2:3]
    
    # Calcular derivadas temporales
    dx_dt = grad(x, t_phys)
    dy_dt = grad(y, t_phys)
    dz_dt = grad(z, t_phys)
    
    # Residuos de las ecuaciones de Lorenz
    f1 = dx_dt - sigma * (y - x)
    f2 = dy_dt - (x * (rho - z) - y)
    f3 = dz_dt - (x * y - beta * z)
    
    # Pérdida PDE (residuo de las ecuaciones)
    loss_pde = MSE(f1, torch.zeros_like(f1)) + \
               MSE(f2, torch.zeros_like(f2)) + \
               MSE(f3, torch.zeros_like(f3))
    
    # Condiciones iniciales
    u0 = PINN(t0)
    x0_pred, y0_pred, z0_pred = u0[:, 0:1], u0[:, 1:2], u0[:, 2:3]
    
    loss_ic = MSE(x0_pred, torch.ones_like(x0_pred) * x0) + \
              MSE(y0_pred, torch.ones_like(y0_pred) * y0) + \
              MSE(z0_pred, torch.ones_like(z0_pred) * z0)
              
    loss_data = MSE(x, x_train) + \
                MSE(y, y_train) + \
                MSE(z, z_train)
    
    # Pérdida total
    return lambda_pde * loss_pde + lambda_ic * loss_ic + 100*loss_data


#==========================================================================
# ETAPA 6: DEFINICIÓN DEl OPTIMIZADOR
#==========================================================================
lr = 0.01
optimizer = pinn_opimizer(lorenz_pinn, lr)

#==========================================================================
# CICLO DE ENTRENAMIENTO
#==========================================================================
training_iter = 50000
loss_values = []

start_time = time.time()

for i in range(training_iter):
    optimizer.zero_grad()
    
    loss = LorenzPINNLoss(lorenz_pinn, t_train, sigma, rho, beta)
    
    loss_values.append(loss.item())
    
    if i % 1000 == 0:
        print(f"Iteration {i}: Loss {loss.item():.10f}")
    
    loss.backward()
    optimizer.step()

elapsed_time = time.time() - start_time
print(f"Training time: {elapsed_time:.2f} seconds")


#==========================================================================
# Validación
#==========================================================================
# Primero, obtén la solución numérica completa para graficar
x_true, y_true, z_true = numerical_sol_lorenz(t_eval, sigma, rho, beta, x0, y0, z0)

# Convertir a numpy para graficar
x_true_np = x_true.detach().cpu().numpy().ravel()
y_true_np = y_true.detach().cpu().numpy().ravel()
z_true_np = z_true.detach().cpu().numpy().ravel()

# Predicciones de la PINN
lorenz_pinn.eval()
with torch.no_grad():
    u_pred = lorenz_pinn(t_eval).cpu().numpy()
    x_pred = u_pred[:, 0]
    y_pred = u_pred[:, 1]
    z_pred = u_pred[:, 2]

t_plot = t_eval.detach().cpu().numpy().ravel()

# Gráfico 1: Series temporales
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(t_plot, x_true_np, 'g-', label='Solución numérica', linewidth=2)
axes[0].plot(t_plot, x_pred, 'r--', label='PINN', linewidth=1.5)
axes[0].set_ylabel('x(t)', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_plot, y_true_np, 'g-', label='Solución numérica', linewidth=2)
axes[1].plot(t_plot, y_pred, 'r--', label='PINN', linewidth=1.5)
axes[1].set_ylabel('y(t)', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_plot, z_true_np, 'g-', label='Solución numérica', linewidth=2)
axes[2].plot(t_plot, z_pred, 'r--', label='PINN', linewidth=1.5)
axes[2].set_xlabel('Tiempo (s)', fontsize=12)
axes[2].set_ylabel('z(t)', fontsize=12)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Gráfico 2: Atractor de Lorenz (3D)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 6))

# Solución numérica
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x_true_np, y_true_np, z_true_np, 'g-', linewidth=0.5, alpha=0.7)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Atractor de Lorenz - Solución Numérica')
ax1.view_init(elev=20, azim=45)  # Ajustar vista

# PINN
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(x_pred, y_pred, z_pred, 'r-', linewidth=0.5, alpha=0.7)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_title('Atractor de Lorenz - PINN')
ax2.view_init(elev=20, azim=45)  # Ajustar vista

plt.tight_layout()
plt.show()

# Gráfico 3: Pérdida durante el entrenamiento
plt.figure(figsize=(10, 5))
plt.plot(loss_values)
plt.xlabel('Iteración')
plt.ylabel('Pérdida')
plt.yscale('log')
plt.title('Convergencia del Entrenamiento')
plt.grid(True, alpha=0.3)
plt.show()

# Gráfico 4: Errores relativos
error_x = torch.norm(torch.tensor(x_pred).view(-1,1) - x_true) / torch.norm(x_true)
error_y = torch.norm(torch.tensor(y_pred).view(-1,1) - y_true) / torch.norm(y_true)
error_z = torch.norm(torch.tensor(z_pred).view(-1,1) - z_true) / torch.norm(z_true)

print(f"\n{'='*50}")
print("MÉTRICAS DE ERROR")
print(f"{'='*50}")
print(f"Error L2 relativo en x: {error_x:.6f}")
print(f"Error L2 relativo en y: {error_y:.6f}")
print(f"Error L2 relativo en z: {error_z:.6f}")
print(f"Error L2 promedio: {(error_x + error_y + error_z)/3:.6f}")
print(f"{'='*50}")