"""
inverse_pinn_quarter.py  (analytic u · optional λ‑fine‑tune)
-----------------------------------------------------------
Recover bending moment M(φ) on 0 … π/2 from a known analytic
ring‑distortion profile u_D(φ).

Physics‑only core:
      u'' + u = -(R²/EI) · M(φ)

Optional data fine‑tune:
      loss  +=  λ · ‖M_pred − M_true‖² ,   λ ∈ [0,1]
        λ = 0   → pure inverse PINN (no M in training)
        λ ≈ 0.05 → gentle snap‑to‑truth after physics convergence

Inputs
------
* φ grid comes from Carranzaet.csv (convenience only).
* Analytic u(φ) & u″(φ) are computed inline.
* M_true(φ) loaded from CSV **for evaluation and optional fine‑tune only**.

Training schedule
-----------------
1. Adam optimiser 8 000 iterations.
2. LBFGS quasi‑Newton polishing 400 iterations.
"""

# ────────── 0. imports ───────────────────────────────────
import numpy as np, pandas as pd, torch, torch.nn as nn, matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch.autograd import grad

# ────────── 1. constants & device ───────────────────────
R, EI = 5.0, 260_416.7           # radius [m], flexural rigidity [kN·m²]
PI    = np.pi
DEV   = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

torch.manual_seed(42); np.random.seed(42)

# ────────── 2. analytic displacement, derivatives & moment ──────

def analytic_ud(phi: np.ndarray) -> np.ndarray:
    a = -0.0009717
    b =  0.0003633
    c =  0.01539
    return a + b * np.cos(phi) + c * np.cos(2 * phi)

def analytic_u_dd(phi: np.ndarray) -> np.ndarray:
    b =  0.0003633
    c =  0.01539
    return -b * np.cos(phi) - 4 * c * np.cos(2 * phi)

def analytic_M(phi: np.ndarray) -> np.ndarray:
    """Analytic bending moment consistent with analytic_ud (kN·m)."""
    return (EI / R**2) * (analytic_ud(phi) + analytic_u_dd(phi))

# ────────── 3. domain grid & ground‑truth M --------------
df      = pd.read_csv('Carranzaet.csv')
phi     = df['phi(radians)'].values.astype(np.float32)   # rad grid 0→π/2
M_true  = df['M'].values.astype(np.float32)              # ground‑truth (for metrics / optional λ)

u_meas  = analytic_ud(phi).astype(np.float32)
u_dd    = analytic_u_dd(phi).astype(np.float32)

# tensors  ( φ̂ = φ/π  ∈ [0,0.5] )
phi_hat = (phi / PI).reshape(-1, 1)
phi_t   = torch.tensor(phi_hat, requires_grad=True , device=DEV, dtype=DTYPE)
u_t     = torch.tensor(u_meas , requires_grad=False, device=DEV, dtype=DTYPE).view(-1,1)
u_dd_t  = torch.tensor(u_dd   , requires_grad=False, device=DEV, dtype=DTYPE).view(-1,1)
M_true_t= torch.tensor(M_true , requires_grad=False, device=DEV, dtype=DTYPE).view(-1,1)

# ────────── 4. neural net that predicts  M(φ) ────────────
class MomentNet(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 5):
        super().__init__()
        seq = [nn.Linear(1, hidden), nn.SiLU()]
        for _ in range(layers - 2):
            seq += [nn.Linear(hidden, hidden), nn.SiLU()]
        seq.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*seq)
    def forward(self, x):
        return self.net(x)

model = MomentNet().to(DEV).type(DTYPE)

# ────────── 5. physics residual ──────────────────────────

def pde_residual(M_pred: torch.Tensor) -> torch.Tensor:
    return u_dd_t + u_t + (R**2 / EI) * (-M_pred)

# data‑fit weight (0 = pure physics)
λ = 0.05   # adjust to 0.05 for light supervision

# ────────── 6. Phase 1 – Adam pre‑training ──────────────
print("\n[Phase 1] Adam optimiser")
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_hist = []

for epoch in range(8000):
    opt.zero_grad(set_to_none=True)
    M_pred = model(phi_t)
    phys_L = torch.mean(pde_residual(M_pred)**2)
    data_L = torch.mean((M_pred - M_true_t)**2)
    loss   = phys_L + λ * data_L
    loss.backward(); opt.step()
    loss_hist.append(loss.item())
    if epoch % 1000 == 0:
        print(f"{epoch:4d} | total={loss.item():.2e} | physics={phys_L.item():.2e}")

# ────────── 6‑bis. Phase 2 – LBFGS polishing ────────────
print("\n[Phase 2] L‑BFGS optimiser")
lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=400,
                          history_size=50, tolerance_grad=1e-9,
                          tolerance_change=1e-11, line_search_fn='strong_wolfe')

def closure() -> torch.Tensor:
    lbfgs.zero_grad(set_to_none=True)
    M_pred = model(phi_t)
    phys_L = torch.mean(pde_residual(M_pred)**2)
    data_L = torch.mean((M_pred - M_true_t)**2)
    loss   = phys_L + λ * data_L
    loss.backward()
    return loss

lbfgs.step(closure)
final_loss = closure().item()
loss_hist.append(final_loss)

# ────────── 7. evaluation & metrics ─────────────────────
M_pred_np = model(phi_t).detach().cpu().numpy().flatten()

mae      = np.mean(np.abs(M_pred_np - M_true))
rmse     = np.sqrt(np.mean((M_pred_np - M_true)**2))
rel_err  = 100 * np.mean(np.abs((M_pred_np - M_true) / M_true))
r2       = r2_score(M_true, M_pred_np)

print('\n=== Evaluation Results ===')
print(f'Final Loss    : {final_loss:.4e}')
print(f'MAE           : {mae:.4e} kN·m')
print(f'RMSE          : {rmse:.4e} kN·m')
print(f'Relative Error: {rel_err:.2f} %')
print(f'R² Score      : {r2:.4f}')

# ────────── 8. plots ────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(phi * 180 / PI, M_true   , 'k-', label='True M')
plt.plot(phi * 180 / PI, M_pred_np, 'r--', label='PINN')
plt.xlabel('angle (deg)'); plt.ylabel('M (kN·m)')
plt.title('Inverse PINN – recovered bending moment'); plt.legend(); plt.grid()

plt.subplot(1, 2, 2)
plt.semilogy(loss_hist)
plt.xlabel('iteration'); plt.ylabel('total loss (log)')
plt.title('Training convergence'); plt.grid()

plt.tight_layout(); plt.savefig('inverse_pinn_quarter.png', dpi=300)
plt.show()

torch.save(model.state_dict(), "inverse_pinn_quarter_state_carranza.pt")