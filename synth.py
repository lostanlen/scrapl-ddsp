import numpy as np
from scipy.signal.windows import tukey
import torch

# duration = 4
# event_duration = 2**(-4)
# sr = 2**13
# fmin = 2**8
# fmax = 2**11
# Q = 24
# hop_length = 2**6
# n_events = 2**6

def get_amplitudes(theta_density, n_events):
    offset = 0.25 * theta_density + 0.75 * theta_density**2
    event_ids = torch.tensor(np.arange(n_events)).type(torch.float32)
    sigmoid_operand = (1-theta_density) * n_events * (event_ids/n_events - offset)
    amplitudes = 1 - torch.sigmoid(2*sigmoid_operand)
    amplitudes = amplitudes / torch.max(amplitudes)
    return amplitudes

def get_slope(theta_slope, Q, hop_length, sr):
    """theta_slope --> ±1 correspond to a near-vertical line.
    theta_slope = 0 corresponds to a horizontal line.
    The output is measured in octaves per second."""
    typical_slope = sr / (Q * hop_length)
    return torch.tan(theta_slope * np.pi/2) * typical_slope/8

def generate_chirp_texture(
    theta_density,
    theta_slope,
    *,
    duration,
    event_duration,
    sr,
    fmin,
    fmax, 
    n_events,
    Q,
    hop_length,
    seed,
):
    # Set random seed
    random_state = np.RandomState(seed)

    # Define constant log(2)
    const_log2 = torch.log(torch.tensor(2.0))

    # Define the amplitudes of the events
    amplitudes = get_amplitudes(theta_density, n_events)

    # Define the slope of the events
    gamma = get_slope(theta_slope, Q, hop_length, sr)

    # Draw onsets at random
    chirp_duration = 2 * event_duration / (torch.abs(theta_slope) + 0.25)
    chirp_length = torch.tensor(chirp_duration * sr).type(torch.int64)
    rand_onsets = random_state.rand(n_events)
    onsets = rand_onsets * (duration*sr/2-chirp_length) + duration * sr / 4
    onsets = torch.floor(onsets).type(torch.int64)

    # Draw frequencies at random
    log2_fmin = torch.log2(torch.tensor(fmin))
    log2_fmax = torch.log2(torch.tensor(fmax))
    rand_pitches = random_state.rand(n_events)
    log2_frequencies = log2_fmin + rand_pitches * (log2_fmax-log2_fmin)
    frequencies = torch.pow(2.0, log2_frequencies)

    # Generate events one by one
    X = torch.zeros(duration*sr, n_events)
    time = torch.arange(chirp_length)/sr - chirp_duration/2
    envelope = torch.tensor(tukey(chirp_length))
    event_ids = torch.arange(n_events)
    patch_zip = zip(event_ids, onsets, amplitudes, frequencies)

    for event_id, onset, amplitude, frequency in patch_zip:
        if torch.abs(gamma) < 1e-6:
            phase = time
        else:
            phase = torch.expm1(gamma*const_log2*time) / (gamma*const_log2)
        chirp = torch.sin(2 * torch.pi * frequency * phase)
        offset = onset + chirp_length
        X[onset:offset, event_id] = chirp * amplitude * envelope

    # Mix events
    x = X.sum(axis=-1)

    return x