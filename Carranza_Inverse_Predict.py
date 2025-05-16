# inverse_pinn_infer.py
import numpy as np
import torch
import torch.nn as nn

# Device & dtype
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE  = torch.float32
PI = np.pi
R  = 5.0
EI = 260_416.7

# ────────── 1. Analytic displacement functions ──────────

def analytic_ud(phi: np.ndarray) -> np.ndarray:
    a = -0.0009717
    b =  0.0003633
    c =  0.01539
    return a + b * np.cos(phi) + c * np.cos(2 * phi)

def analytic_u_dd(phi: np.ndarray) -> np.ndarray:
    b =  0.0003633
    c =  0.01539
    return -b * np.cos(phi) - 4 * c * np.cos(2 * phi)

# ────────── 2. Network definition (same as training) ────

class MomentNet(nn.Module):
    def __init__(self, hidden=64, layers=5):
        super().__init__()
        seq = [nn.Linear(1, hidden), nn.SiLU()]
        for _ in range(layers - 2):
            seq += [nn.Linear(hidden, hidden), nn.SiLU()]
        seq.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*seq)
    def forward(self, x): return self.net(x)

# ────────── 3. Load trained model ───────────────────────

model = MomentNet().to(DEVICE).type(DTYPE)
model.load_state_dict(torch.load("inverse_pinn_quarter_state_carranza.pt", map_location=DEVICE))
model.eval()

# ────────── 4. Prediction function ──────────────────────

def predict_moment(phi_deg: float | list | np.ndarray) -> np.ndarray:
    """Predict M(φ) in kN·m from angle in degrees using trained PINN."""
    phi_arr = np.atleast_1d(phi_deg).astype(np.float32)
    phi_rad = np.deg2rad(phi_arr)
    phi_hat = (phi_rad / PI).reshape(-1, 1)

    u      = analytic_ud(phi_rad).reshape(-1, 1)
    u_dd   = analytic_u_dd(phi_rad).reshape(-1, 1)

    phi_t     = torch.tensor(phi_hat, dtype=DTYPE, device=DEVICE)
    u_t       = torch.tensor(u     , dtype=DTYPE, device=DEVICE)
    u_dd_t    = torch.tensor(u_dd  , dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        M_pred = model(phi_t).cpu().numpy().flatten()
    return M_pred if phi_arr.size > 1 else M_pred[0]

# ────────── 5. Demo ─────────────────────────────────────

if __name__ == "__main__":
    test_phi_deg = [0, 15, 30, 45, 60, 75, 90]
    M_out = predict_moment(test_phi_deg)

    print("φ (deg) | M_pred (kN·m)")
    for φ, m in zip(test_phi_deg, M_out):
        print(f"{φ:6.1f} | {m:+.6f}")
