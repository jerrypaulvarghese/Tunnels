# Tunnel Distortion PINN Models (Carranza-Torres)

This repository contains two Physics-Informed Neural Network (PINN) implementations:

- **Forward PINN**: Predicts ring distortion displacement \( u_D(\phi) \) given a known bending moment field \( M(\phi) \)
- **Inverse PINN**: Recovers the bending moment \( M(\phi) \) from known displacement \( u_D(\phi) \)

Both models are trained on the quarter-ring section \( \phi \in [0, \pi/2] \), representing the upper-left quadrant of a circular tunnel lining under soil load.

---

## 🔁 Governing Equation

The base PDE used in both models:

\[
u''(\phi) + u(\phi) = -\left(\frac{R^2}{EI}\right) M(\phi)
\]

Where:
- \( R = 5.0 \) m (radius)
- \( EI = 260416.7 \) kN·m² (flexural rigidity)
- \( \phi \in [0, \pi/2] \) (angle in radians)

---

## 📁 Files Included

| File | Description |
|------|-------------|
| `Carranzaet.csv` | Input CSV with φ, displacement \( u_D \), and moment \( M \) |
| `Carranza_Forward_PINN.py` | Train forward PINN using analytic M(φ) |
| `Carranza_Inverse_PINN.py` | Train inverse PINN using analytic u(φ) |
| `Carranza_forward_Predict.py` | Use trained forward model to infer \( u_D \) |
| `Carranza_Inverse_Predict.py` | Use trained inverse model to infer \( M \) |
| `forward_pinn_quarter_state_carranza.pt` | Trained forward model |
| `inverse_pinn_quarter_state_carranza.pt` | Trained inverse model |
| `*.png` | Plots of predictions vs ground truth |
| `.gitignore` | Ignores `venv/`, cache, etc. |

---

## 🔧 Training Logic

### Forward PINN
- Trained with known **analytic** \( M(\phi) \)
- Learns to map φ → \( u_D(\phi) \)
- Boundary Conditions: \( u(0) \), \( u(\pi/2) \) from analytic solution

### Inverse PINN
- Trained using **analytic** \( u(\phi) \), computes \( u''(\phi) \)
- Learns to map φ → \( M(\phi) \)
- Can optionally fine-tune with true \( M \) using supervision weight \( \lambda \)

---

## 🧪 Inference

To infer predictions:
- Use `Carranza_forward_Predict.py` → input angle φ → get \( u_D(\phi) \)
- Use `Carranza_Inverse_Predict.py` → input angle φ → get \( M(\phi) \)

---

## 📊 Visuals

Prediction comparisons are saved as:
- `forward_pinn_quarter.png`
- `inverse_pinn_quarter.png`

Each plot shows:
- Ground truth vs predicted curve
- Training loss convergence

---

## 📦 Requirements

Install with:

pip install -r requirements.txt

