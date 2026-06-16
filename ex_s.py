import numpy as np
import matplotlib.pyplot as plt


def mother_mexican_hat(t):
    return (1 - t**2) * np.exp(-t**2 / 2)

def coefficients_from_scratch(signal, scales, span=10.0):

    signal = np.array(signal, dtype=float)
    n_samples = len(signal)

    coeffs = np.zeros((len(scales), n_samples))
    
    for i, scale in enumerate(scales):

        n_points = int(span * scale) * 2 + 1
        t = np.linspace(-span * scale, span * scale, n_points)
        scaled = mother_mexican_hat(t / scale)
        scaled = scaled / np.sqrt(scale)
        coeffs[i, :] = np.convolve(signal, scaled[::-1], mode='same')
        
    return coeffs




def inverse_scratch(coeffs, scales, dx=1.0):

    n_scales, n_samples = coeffs.shape
    reconstructed = np.zeros(n_samples)
    

    for idx, scale in enumerate(scales):

        reconstructed += coeffs[idx, :] / (scale ** 1.5)

    factor = 1.45 
    reconstructed *= (dx / factor)
    
    return reconstructed


dim=200

np.random.seed(42)
t_signal = np.linspace(0, dim/5, dim)

true_signal = 3.2*np.cos(np.pi*t_signal/8+np.pi/3) + 1.5 * np.sin(2*np.pi*t_signal+np.pi/8) 
noisy_signal = true_signal + 0.0015 * np.random.randn(dim)


scales = np.arange(1, dim/20, dtype=float)
coefficients=coefficients_from_scratch(noisy_signal, scales, span=10.0)



reconstructed_signal = inverse_scratch(coefficients, scales)


fig, axes = plt.subplots(3, 1, figsize=(10, 10))


axes[0].plot(t_signal, noisy_signal, color='gray', alpha=0.6, label='Noisy Input')
axes[0].plot(t_signal, true_signal, color='black', lw=2, label='True Signal')
axes[0].set_title("signal")
axes[0].legend()

im = axes[1].imshow(coefficients, aspect='auto', cmap='jet', 
                     extent=[t_signal[0], t_signal[-1], scales[-1], scales[0]])
axes[1].set_title(" spectrum (Scalogram View)")
axes[1].set_ylabel("Scales (Width)")
fig.colorbar(im, ax=axes[1], label='Intensity')


axes[2].plot(t_signal, reconstructed_signal, color='blue', lw=2, label='denoize or approximated signal')
axes[2].set_title("approximation")
axes[2].legend()

for ax in [axes[0], axes[2]]:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
