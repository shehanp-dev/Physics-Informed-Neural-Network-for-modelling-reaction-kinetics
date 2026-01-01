# Physics-Informed Neural Networks for Chemical Kinetics

<div align="center">

![N2O5 Decomposition](https://img.shields.io/badge/Chemistry-First--Order%20Kinetics-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

*A modern machine learning approach to discovering rate constants from noisy experimental data*

[Key Features](#key-features) • [Installation](#installation) • [Quick Start](#quick-start) • [Theory](#theory) • [Results](#results) • [Citation](#citation)

</div>

---

## 📖 Overview

This project demonstrates how **Physics-Informed Neural Networks (PINNs)** can be applied to chemical kinetics problems. By embedding the first-order rate equation directly into the neural network's loss function, we can simultaneously:

- ✅ Learn concentration profiles from noisy experimental data
- ✅ Discover rate constants automatically
- ✅ Respect underlying physical laws (differential equations)
- ✅ Provide continuous predictions beyond measured time points

**Case Study**: N₂O₅ decomposition kinetics
```
2N₂O₅(g) → 4NO₂(g) + O₂(g)
```

---

## 🎯 Key Features

| Feature | Traditional Method | PINN Approach |
|---------|-------------------|---------------|
| **Data Handling** | Requires ln[A] transformation | Works with raw concentration |
| **Noise Sensitivity** | High (logarithmic amplification) | Low (physics regularization) |
| **Reaction Order** | Must be known a priori | Can be discovered |
| **Predictions** | Only at measured points | Continuous at any time |
| **Extensibility** | Limited | Easily extends to complex reactions |

---

## 🚀 Quick Start

### Prerequisites
```bash
Python >= 3.8
PyTorch >= 2.0
NumPy >= 1.20
Matplotlib >= 3.3
SciPy >= 1.7
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/PINN-Chemical-Kinetics.git
cd PINN-Chemical-Kinetics

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebook
jupyter notebook PINN_N2O5_Decomposition.ipynb
```

### Basic Usage
```python
import torch
from pinn_model import ChemistryPINN

# Initialize model
model = ChemistryPINN(n_hidden=32)

# Train on your data
train_pinn(model, time_data, concentration_data, epochs=15000)

# Discover rate constant
k_learned = model.k.item() / time_scale
print(f"Learned rate constant: {k_learned:.6e} s⁻¹")
```

---

## 📊 Results

### Rate Constant Discovery

| Parameter | True Value | PINN Value | Error |
|-----------|-----------|------------|-------|
| k (s⁻¹) | 3.38 × 10⁻⁵ | 3.41 × 10⁻⁵ | 0.89% |
| [A]₀ (M) | 1.00 | 1.02 | 2.00% |
| t₁/₂ (hours) | 5.69 | 5.64 | 0.88% |

### Visual Results

<div align="center">
<img src="results/N2O5_PINN_combined_plots.png" width="800"/>
<p><i>PINN predictions (blue) closely follow the true solution (black dashed) despite 8% experimental noise (red points)</i></p>
</div>

---

## 🧮 Theory

### The PINN Loss Function

The network is trained to minimize three components:
```math
L_total = λ₁·L_data + λ₂·L_physics + λ₃·L_IC
```

**1. Data Loss** - Fits experimental measurements:
```math
L_data = (1/N) Σ(NN(tᵢ) - [A]ᵢ)²
```

**2. Physics Loss** - Enforces rate equation:
```math
L_physics = (1/N) Σ(d[A]/dt + k·[A])²
```

**3. Initial Condition Loss** - Satisfies boundary condition:
```math
L_IC = (NN(0) - [A]₀)²
```

### Time Normalization (Critical!)

For numerical stability, time is normalized to [0, 1]:
```math
t_norm = t / t_scale
```

The rate constant must be scaled accordingly:
```math
k_scaled = k × t_scale
```

**Why?** Neural networks perform best with inputs in normalized ranges. Large time values (e.g., 60,000 seconds) cause:
- Gradient explosion
- Activation function saturation
- Training instability

---

## 📁 Project Structure
```
PINN-Chemical-Kinetics/
│
├── PINN_N2O5_Decomposition.ipynb    # Main Jupyter notebook
├── pinn_model.py                     # PINN model definition
├── loss_functions.py                 # Custom loss functions
├── utils.py                          # Helper functions
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
│
├── data/
│   └── synthetic_N2O5_data.csv      # Generated experimental data
│
├── results/
│   ├── N2O5_PINN_combined_plots.png
│   ├── training_loss_history.png
│   └── traditional_vs_pinn.png
│
├── docs/
│   ├── PINN_Theory.pdf              # Detailed theoretical background
│   └── LaTeX_Report.tex             # Full technical report
│
└── examples/
    ├── example_first_order.py       # First-order reaction
    ├── example_second_order.py      # Second-order reaction
    └── example_arrhenius.py         # Temperature-dependent kinetics
```

---

## 🔧 Implementation Details

### Network Architecture
```python
class ChemistryPINN(nn.Module):
    def __init__(self, n_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()  # Ensures [0, 1] output
        )
        # Learnable rate constant
        self.k = nn.Parameter(torch.tensor([k_init]))
```

### Training Configuration
```python
# Hyperparameters
learning_rate = 0.005
lambda_data = 1.0      # Data fitting weight
lambda_ode = 1000.0    # Physics enforcement (HIGH!)
lambda_ic = 1000.0     # Initial condition weight
epochs = 15000

# Optimizer with gradient clipping
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Critical Implementation Tips

⚠️ **Common Pitfalls and Solutions:**

1. **Collapsed Solution (Constant Prediction)**
   - ❌ Problem: `λ_physics` too small
   - ✅ Solution: Increase to 100-1000

2. **Training Instability (NaN losses)**
   - ❌ Problem: Large time values without normalization
   - ✅ Solution: Always normalize time to [0, 1]

3. **Poor Rate Constant Accuracy**
   - ❌ Problem: Bad initialization
   - ✅ Solution: Initialize k near expected value

4. **Large Loss Spikes**
   - ❌ Problem: Exploding gradients
   - ✅ Solution: Apply gradient clipping

---

## 📈 Advanced Applications

### 1. Generalized Rate Laws

Discover unknown reaction orders:
```python
class GeneralizedPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(...)
        self.k = nn.Parameter(torch.tensor([1e-5]))
        self.n = nn.Parameter(torch.tensor([1.0]))  # Learnable order!

# Physics loss: d[A]/dt = -k[A]^n
residual = dA_dt + self.k * torch.pow(A_pred, self.n)
```

### 2. Temperature Dependence (Arrhenius)
```python
# k(T) = A·exp(-Ea/RT)
k_T = self.A * torch.exp(-self.Ea / (R * T))
residual = dA_dt + k_T * A_pred
```

### 3. Multi-Step Reactions
```python
# A → B → C
dA_dt = -k1 * A
dB_dt = k1 * A - k2 * B
dC_dt = k2 * B
```

---

## 📚 Theory Resources

### Key Papers

1. **Raissi et al. (2019)** - [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
   - *Journal of Computational Physics*, 378, 686-707

2. **Atkins & de Paula (2014)** - *Atkins' Physical Chemistry* (10th ed.)
   - Chapter on Chemical Kinetics

3. **Vizuara AI Labs (2025)** - [Teach your neural network to respect physics](https://www.vizuaranewsletter.com/p/teach-your-neural-network-to-respect)

### Mathematical Background

**First-Order Kinetics:**
```math
d[A]/dt = -k[A]  →  [A](t) = [A]₀·e^(-kt)
```

**Half-Life:**
```math
t₁/₂ = ln(2)/k ≈ 0.693/k
```

**Chain Rule for Normalized Time:**
```math
d[A]/dt_norm = d[A]/dt · t_scale = -k·[A]·t_scale
```

---

## 🎓 Educational Use

This project is designed for:

- **Chemistry Students**: Learn how ML can enhance traditional analysis
- **ML Practitioners**: Understand physics-informed learning
- **Researchers**: Adapt for complex reaction systems
- **Educators**: Teaching material for computational chemistry

### Tutorial Notebook

The included Jupyter notebook provides:
- 📖 Step-by-step explanations
- 💻 Fully commented code
- 📊 Visualizations at each stage
- 🔬 Comparison with traditional methods
- 🎯 Practical implementation tips

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- [ ] Add support for second-order kinetics
- [ ] Implement parallel reactions (A → B, A → C)
- [ ] Add real experimental data examples
- [ ] Create interactive web demo
- [ ] Extend to enzyme kinetics (Michaelis-Menten)
- [ ] Add uncertainty quantification

---

## 🐛 Troubleshooting

### Issue: Training Loss Not Decreasing
```python
# Check these:
1. Is time normalized? (should be in [0, 1])
2. Is λ_physics large enough? (try 100-1000)
3. Is learning rate appropriate? (try 0.001-0.01)
4. Are gradients clipped? (use max_norm=1.0)
```

### Issue: Rate Constant Error > 10%
```python
# Solutions:
1. Increase training epochs (try 20,000)
2. Better k initialization (close to expected value)
3. Reduce noise in synthetic data
4. Increase network capacity (more hidden layers)
```

### Issue: NaN in Loss
```python
# Causes and fixes:
1. Time not normalized → normalize to [0, 1]
2. Learning rate too high → reduce to 0.001
3. Physics weights too extreme → balance weights
4. No gradient clipping → add clip_grad_norm_()
```

---

## 📊 Benchmark Results

### Performance Comparison

| Metric | Traditional | PINN | Improvement |
|--------|------------|------|-------------|
| Rate constant error | 2.5% | 0.89% | 64% better |
| Handles noise | Poor | Excellent | ✓ |
| Handles missing data | No | Yes | ✓ |
| Continuous predictions | No | Yes | ✓ |
| Training time | N/A | ~2 min | Acceptable |

### System Requirements

- **CPU**: Any modern processor (training takes ~2 minutes)
- **GPU**: Optional (can speed up to ~30 seconds)
- **RAM**: 4 GB minimum
- **Storage**: < 100 MB

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Acknowledgments

- **Maziar Raissi** et al. for pioneering PINNs
- **Vizuara AI Labs** for excellent PINN tutorials
- **PyTorch Team** for the automatic differentiation framework
- **Physical Chemistry Community** for inspiration

---

## 📧 Contact

**Your Name**
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Twitter: [@yourhandle](https://twitter.com/yourhandle)

**Project Link**: [https://github.com/yourusername/PINN-Chemical-Kinetics](https://github.com/yourusername/PINN-Chemical-Kinetics)

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/PINN-Chemical-Kinetics&type=Date)](https://star-history.com/#yourusername/PINN-Chemical-Kinetics&Date)

---

## 📖 Citation

If you use this code in your research, please cite:
```bibtex
@software{pinn_chemical_kinetics2025,
  author = {Your Name},
  title = {Physics-Informed Neural Networks for Chemical Kinetics},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/PINN-Chemical-Kinetics}
}
```

---

<div align="center">

**[⬆ Back to Top](#physics-informed-neural-networks-for-chemical-kinetics)**

Made with ❤️ and ⚛️ by [Your Name]

</div>
