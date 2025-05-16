"""
forward_pinn_quarter.py  (analytic M · optional λ‑fine‑tune)
-----------------------------------------------------------
Predict distortion displacement u_D(φ) on 0 … π/2 from a **known
analytic bending‑moment curve** M(φ).

Governing PDE:
      u'' + u = -(R²/EI) · M(φ)

Optional data fine‑tune:
      loss += λ · ‖u_pred − u_true‖² ,   λ ∈ [0,1]
        λ = 0   → pure forward PINN (no displacement data in loss)
        λ ≈ 0.05 → gentle snap‑to‑truth after physics convergence

Inputs
------
* φ grid comes from Carranzaet.csv (only for convenience).
* Analytic M(φ) is computed inline.
* u_true(φ) from CSV is **used only for evaluation and optional fine‑tune**.

Training schedule
-----------------
1. Adam optimiser 8 000 iterations.
2. LBFGS quasi‑Newton polishing 400 iterations.
"""

# ────────── 0. imports & reproducibility ──────────────────────────
import numpy as np, pandas as pd, torch, torch.nn as nn, matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch.autograd import grad

torch.manual_seed(42); np.random.seed(42)
DEV   = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

# ────────── 1. constants ─────────────────────────────────────────
R  = 5.0          # m
EI = 260_416.7    # kN·m²
PI = np.pi

# ────────── 2. analytic M(φ) ------------------------------------

def analytic_M(phi: np.ndarray) -> np.ndarray:
    """Closed‑form bending moment (kN·m)."""
    # coefficients from Carranza‑Torres solution
    a = -0.0009717
    b =  0.0003633
    c =  0.01539
    u   = a + b * np.cos(phi) + c * np.cos(2 * phi)
    u_dd= -b * np.cos(phi) - 4 * c * np.cos(2 * phi)
    return (EI / R**2) * (u + u_dd)

# ────────── 3. domain grid & ground‑truth u ----------------------
df     = pd.read_csv('Carranzaet.csv')
phi    = df['phi(radians)'].values.astype(np.float32)   # rad 0→π/2
u_true = df['u'].values.astype(np.float32)              # displacement for metrics / λ
M_known= analytic_M(phi).astype(np.float32)

# tensors (φ̂ = φ/π ∈ [0,0.5])
phi_hat = (phi / PI).reshape(-1, 1)
phi_t   = torch.tensor(phi_hat, requires_grad=True, device=DEV, dtype=DTYPE)
M_t     = torch.tensor(M_known , requires_grad=False, device=DEV, dtype=DTYPE).view(-1,1)
u_true_t= torch.tensor(u_true  , requires_grad=False, device=DEV, dtype=DTYPE).view(-1,1)

# ────────── 4. neural net that predicts u(φ) ---------------------
class DisplacementNet(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 5):
        super().__init__()
        seq = [nn.Linear(1, hidden), nn.SiLU()]
        for _ in range(layers - 2):
            seq += [nn.Linear(hidden, hidden), nn.SiLU()]
        seq.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*seq)
    def forward(self, x):
        return self.net(x)

model = DisplacementNet().to(DEV).type(DTYPE)

# ────────── 5. physics residual (π‑scaling & sign) ---------------

def pde_residual(phi_hat: torch.Tensor, u_pred: torch.Tensor, M_known: torch.Tensor) -> torch.Tensor:
    du_hat  = grad(u_pred, phi_hat, torch.ones_like(u_pred), create_graph=True)[0]
    d2u_hat = grad(du_hat , phi_hat, torch.ones_like(du_hat), create_graph=True)[0]
    du  = du_hat  / PI
    d2u = d2u_hat / PI**2
    return d2u + u_pred + (R**2 / EI) * (-M_known)

# boundary conditions from analytic solution
u0  = torch.tensor([[u_true[0]]],  device=DEV, dtype=DTYPE)
u90 = torch.tensor([[u_true[-1]]], device=DEV, dtype=DTYPE)
phi_bc = torch.tensor([[0.0], [0.5]], device=DEV, dtype=DTYPE)

def bc_penalty() -> torch.Tensor:
    u_bc = model(phi_bc)
    return (u_bc[0]-u0).pow(2).mean() + (u_bc[1]-u90).pow(2).mean()

W_BC = 10.0

# data‑fit weight (0 = pure physics)
λ = 0.0   # set to 0.05 for light supervision

# ────────── 6. Phase 1 – Adam pre‑training ------------------------
print("\n[Phase 1] Adam optimiser")
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.5, patience=200)
loss_hist = []

for epoch in range(8000):
    opt.zero_grad(set_to_none=True)
    u_pred = model(phi_t)
    phys_L = pde_residual(phi_t, u_pred, M_t).pow(2).mean()
    data_L = torch.mean((u_pred - u_true_t)**2)
    loss   = phys_L + W_BC * bc_penalty() + λ * data_L
    loss.backward(); opt.step(); sched.step(loss)
    loss_hist.append(loss.item())
    if epoch % 1000 == 0:
        print(f"{epoch:4d} | total={loss.item():.2e} | physics={phys_L.item():.2e}")

# ────────── 6‑bis. Phase 2 – LBFGS polishing ----------------------
print("\n[Phase 2] L‑BFGS optimiser")
lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=400,
                          history_size=50, tolerance_grad=1e-9,
                          tolerance_change=1e-11, line_search_fn='strong_wolfe')

def closure() -> torch.Tensor:
    lbfgs.zero_grad(set_to_none=True)
    u_pred = model(phi_t)
    phys_L = pde_residual(phi_t, u_pred, M_t).pow(2).mean()
    data_L = torch.mean((u_pred - u_true_t)**2)
    loss   = phys_L + W_BC * bc_penalty() + λ * data_L
    loss.backward()
    return loss

lbfgs.step(closure)
final_loss = closure().item()
loss_hist.append(final_loss)

# ────────── 7. evaluation & metrics -------------------------------
u_pred_np = model(phi_t).detach().cpu().numpy().flatten()

mae      = np.mean(np.abs(u_pred_np - u_true))
rmse     = np.sqrt(np.mean((u_pred_np - u_true)**2))
rel_err  = 100 * np.mean(np.abs((u_pred_np - u_true) / u_true))
r2       = r2_score(u_true, u_pred_np)

print('\n=== Evaluation Results ===')
print(f'Final Loss    : {final_loss:.4e}')
print(f'MAE           : {mae:.4e} m')
print(f'RMSE          : {rmse:.4e} m')
print(f'Relative Error: {rel_err:.2f} %')
print(f'R² Score      : {r2:.4f}')

# ────────── 8. plots ------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(phi * 180 / PI, u_true   , 'k-', label='True u')
plt.plot(phi * 180 / PI, u_pred_np, 'r--', label='PINN')
plt.xlabel('angle (deg)'); plt.ylabel('u_D (m)')
plt.title('Forward PINN – predicted displacement'); plt.legend(); plt.grid()

plt.subplot(1, 2, 2)
plt.semilogy(loss_hist)
plt.xlabel('iteration'); plt.ylabel('total loss (log)')
plt.title('Training convergence'); plt.grid()

plt.tight_layout(); plt.savefig('forward_pinn_quarter.png', dpi=300)
plt.show()

torch.save(model.state_dict(), "forward_pinn_quarter_state_carranza.pt")
