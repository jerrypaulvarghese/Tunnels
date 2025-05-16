# forward_pinn_infer.py
import numpy as np
import torch
import torch.nn as nn

# Device setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32
PI = np.pi

# Constants from training
R = 5.0
EI = 260_416.7

# Redefine network structure (must match training)
class DisplacementNet(nn.Module):
    def __init__(self, hidden=64, layers=5):
        super().__init__()
        seq = [nn.Linear(1, hidden), nn.SiLU()]
        for _ in range(layers - 2):
            seq += [nn.Linear(hidden, hidden), nn.SiLU()]
        seq.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*seq)
    def forward(self, x):
        return self.net(x)

# Load trained model
model = DisplacementNet().to(DEVICE).type(DTYPE)
model.load_state_dict(torch.load("forward_pinn_quarter_state_carranza.pt", map_location=DEVICE))
model.eval()

# Prediction helper
def predict_displacement(phi_deg: float | list | np.ndarray) -> np.ndarray:
    phi_arr = np.atleast_1d(phi_deg).astype(np.float32)
    phi_hat = (np.deg2rad(phi_arr) / PI).reshape(-1, 1)  # φ̂ = φ / π
    with torch.no_grad():
        phi_tensor = torch.tensor(phi_hat, dtype=DTYPE, device=DEVICE)
        u_pred = model(phi_tensor).cpu().numpy().flatten()
    return u_pred if phi_arr.size > 1 else u_pred[0]

# Demo
if __name__ == "__main__":
    input_phi_deg = [0, 15, 30, 45, 60, 75, 90]  # can change this
    u_pred = predict_displacement(input_phi_deg)

    print("φ (deg) | u_D (m)")
    for phi, u in zip(input_phi_deg, u_pred):
        print(f"{phi:6.1f} | {u:+.6e}")
