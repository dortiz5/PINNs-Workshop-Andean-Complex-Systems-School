import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib as mlp

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

# Configuración
torch.manual_seed(42)
np.random.seed(42)

# ===== 1. GENERAR DATOS =====
# Función verdadera: u(t) = sin(t)
def true_function(t):
    return np.sin(t)

# Derivada: du/dt = cos(t)
def true_derivative(t):
    return np.cos(t)

# Datos de entrenamiento
t_train = np.sort(np.random.uniform(0, 12, 50))
u_train = true_function(t_train) + np.random.normal(0, 0.02, len(t_train))

# Datos de test
t_test = np.sort(np.random.uniform(0, 12, 30))
u_test = true_function(t_test) + np.random.normal(0, 0.02, len(t_test))

# Convertir a tensores
t_train_tensor = torch.FloatTensor(t_train.reshape(-1, 1))
u_train_tensor = torch.FloatTensor(u_train.reshape(-1, 1))
t_test_tensor = torch.FloatTensor(t_test.reshape(-1, 1))
u_test_tensor = torch.FloatTensor(u_test.reshape(-1, 1))

# ===== 2. DEFINIR RED NEURONAL =====
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

# ===== 3. ENTRENAR RED =====
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 2000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(t_train_tensor)
    loss = criterion(outputs, u_train_tensor)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

# ===== 4. PREDICCIONES =====
model.eval()
with torch.no_grad():
    t_continuous = np.linspace(0, 12, 1000)
    t_continuous_tensor = torch.FloatTensor(t_continuous.reshape(-1, 1))
    u_nn = model(t_continuous_tensor).numpy().flatten()
    
    u_train_pred = model(t_train_tensor).numpy().flatten()
    u_test_pred = model(t_test_tensor).numpy().flatten()

# ===== 5. CALCULAR ENERGÍA =====
# Energía total: E = 0.5 * (du/dt)^2 + 0.5 * u^2

def compute_energy_numerical(t, u):
    """Calcula energía total usando derivadas numéricas"""
    dt = t[1] - t[0]
    dudt = np.gradient(u, dt)
    energy_kinetic = 0.5 * dudt**2
    energy_potential = 0.5 * u**2
    energy_total = energy_kinetic + energy_potential
    return energy_total

def compute_energy_nn(model, t):
    """Calcula energía total usando autograd de PyTorch"""
    t_tensor = torch.FloatTensor(t.reshape(-1, 1))
    t_tensor.requires_grad = True
    
    u = model(t_tensor)
    
    # Calcular du/dt usando autograd
    dudt = torch.autograd.grad(u, t_tensor, 
                                grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
    
    # Energía cinética + potencial
    energy_kinetic = 0.5 * dudt**2
    energy_potential = 0.5 * u**2
    energy_total = energy_kinetic + energy_potential
    
    return energy_total.detach().numpy().flatten()

# Calcular energías
t_energy = np.linspace(0, 10, 1000)
energy_real = compute_energy_numerical(t_energy, true_function(t_energy))
energy_nn = compute_energy_nn(model, t_energy)

# ===== 6. GRAFICAR =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Displacement
ax1.plot(t_continuous, true_function(t_continuous), 'g-', 
         linewidth=2, label='Solución Real', alpha=0.7)
ax1.scatter(t_train, u_train, c='blue', s=30, alpha=0.6, 
           label='conjunto entrenamiento', marker='*')
ax1.scatter(t_test, u_test, c='red', s=30, alpha=0.6, 
           label='conjunto test', marker='.')
ax1.plot(t_continuous, u_nn, 'cyan', linewidth=1.5, 
         alpha=0.8, label='Solución NN')

ax1.set_xlabel('t', fontsize=12)
ax1.set_ylabel('u', fontsize=12)
ax1.set_title('Desplazamiento u vs tiempo', fontsize=13)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 12)

# Gráfico 2: Energía
ax2.plot(t_energy, energy_nn, 'g-', linewidth=2, 
         label='Energía de la Red', alpha=0.8)
ax2.plot(t_energy, energy_real, 'gray', linewidth=2, 
         label='Energía Real', linestyle='-', alpha=0.7)

ax2.set_xlabel('t', fontsize=12)
ax2.set_ylabel('Energía', fontsize=12)
ax2.set_title('Energía de la Red Neuronal y Energía Real', fontsize=13)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 10)

plt.tight_layout()
plt.savefig('resultados_red_neuronal.pdf', dpi=300, bbox_inches='tight')
plt.show()


# ===== 7. MÉTRICAS =====
print("\n" + "="*50)
print("MÉTRICAS DE RENDIMIENTO")
print("="*50)
print(f"MSE Train: {np.mean((u_train - u_train_pred)**2):.6f}")
print(f"MSE Test: {np.mean((u_test - u_test_pred)**2):.6f}")
print(f"Error energía promedio: {np.mean(np.abs(energy_nn - energy_real)):.6f}")
print("="*50)