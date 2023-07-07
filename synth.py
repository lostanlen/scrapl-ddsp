import numpy as np
from scipy.signal.windows import tukey
import torch

# duration = 4
# sr = 2**13
# fmin = 2**8
# fmax = 2**9
# n_events = 128
# threshold_db = -10

def generate_chirp_texture(
        gamma,
        log2_density,
        *,
        duration,
        event_duration,
        sr,
        fmin,
        fmax, 
        n_events,
        threshold_db):

    # Enable automatic differentiation for gamma and log2_density
    gamma = torch.tensor(gamma, requires_grad=True)
    log2_density = torch.tensor(log2_density, requires_grad=True)

    # Define the amplitudes of the events
    event_ids = torch.tensor(np.arange(n_events)).type(torch.int64)
    density = torch.pow(2.0, log2_density)
    amplitudes_db = event_ids * threshold_db / density
    amplitudes = torch.nn.functional.softmax(amplitudes_db, dim=-1)

    # Draw onsets at random
    event_length = torch.tensor(event_duration * sr).type(torch.int64)
    onsets = torch.floor(torch.rand(n_events) * (duration*sr - event_length))
    onsets = onsets.type(torch.int64)

    # Draw frequencies at random
    log2_fmin = torch.log2(torch.tensor(fmin))
    log2_fmax = torch.log2(torch.tensor(fmax))
    log2_frequencies = log2_fmin + torch.rand(n_events) * (log2_fmax-log2_fmin)
    frequencies = torch.pow(2.0, log2_frequencies)

    # Generate events one by one
    X = torch.zeros(duration*sr, n_events)
    time = torch.arange(0, event_duration, 1/sr)
    envelope = torch.tensor(tukey(event_length))
    patch_zip = zip(event_ids, onsets, amplitudes, frequencies)
    for event_id, onset, amplitude, frequency in patch_zip:
        chirp_phase = torch.expm1(gamma*torch.log(2)*time) / (gamma*torch.log(2))
        chirp = torch.sin(frequency * chirp_phase)
        offset = onset + event_length
        X[onset:offset, event_id] = chirp * amplitude * envelope

    # Mix events
    x = X.sum(axis=-1)

    # Apply a Tukey window onto the mixture
    window = torch.tensor(tukey(duration*sr))
    y = x * window

    return y