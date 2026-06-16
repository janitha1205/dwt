import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal




dim=2000
np.random.seed(42)
t = np.linspace(0, dim/5, dim)

true_signal = 3.2*np.cos(np.pi*t/8+np.pi/3) + 1.5 * np.sin(2*np.pi*t+np.pi/8) 
noisy_signal = true_signal + 1.15 * np.random.randn(dim)

scales = np.arange(1, dim/20, dtype=float)




coefficients = signal.cwt(noisy_signal, signal.ricker, scales)


reconstructed_signal = np.sum(coefficients / (scales[:, None] ** 1.5), axis=0)

c_psi_factor = 1.45  
reconstructed_signal *= (1.0 / c_psi_factor)


fig, axes = plt.subplots(3, 1, figsize=(10, 10))


axes[0].plot(t, noisy_signal, color='gray', alpha=0.6, label='Noisy Input')
axes[0].plot(t, true_signal, color='black', lw=2, label='True Signal')
axes[0].set_title("Signal with present of noise/not noise")
axes[0].legend()


im = axes[1].imshow(coefficients, aspect='auto', cmap='jet', 
                     extent=[t[0], t[-1], scales[-1], scales[0]])
axes[1].set_title("Coefficients Matrix (Scalogram View)")
axes[1].set_ylabel("Scales")
fig.colorbar(im, ax=axes[1], label='Intensity')

axes[2].plot(t, reconstructed_signal, color='blue', lw=2, label='Reconstructed via Summation')
axes[2].set_title("3. approximated Signal ")
axes[2].legend()

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
