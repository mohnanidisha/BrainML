"""
BrainML DEMO: Cognitive workload detection from simulated fNIRS
Classes: 0-back (easy), 2-back (medium), 3-back (hard)

Pipeline: simulate -> bandpass filter -> epoch + baseline -> features
          -> Leave-One-Subject-Out CV (SVM vs Random Forest) -> plots

Run in Google Colab or locally:
    pip install numpy scipy scikit-learn matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.stats import gamma, skew
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

rng = np.random.default_rng(42)

# ------------------------- 1. SETTINGS -----------------------------------
fs = 10                      # sampling rate (Hz)
n_subjects = 12
n_channels = 8               # prefrontal channels
trials_per_class = 10
class_names = ["0-back", "2-back", "3-back"]
level_gain = np.array([0.3, 1.0, 1.5])   # workload -> response size
trial_dur = 30               # seconds between trial onsets
task_dur = 15                # seconds of task per trial
epoch_len = 20               # epoch length (s)
base_len = 2                 # baseline window (s)
noise_level = 3.0            # raise to make the task harder
subject_norm = True          # z-score features within each subject

n_classes = len(class_names)
n_trials = n_classes * trials_per_class
T = n_trials * trial_dur + 20
t = np.arange(0, T, 1 / fs)
N = len(t)

# ------------------------- 2. SIMULATE DATA ------------------------------
# Hemodynamic response function (double gamma), normalised to peak 1
hrf_t = np.arange(0, 30, 1 / fs)
hrf = gamma.pdf(hrf_t, 6, scale=1) - gamma.pdf(hrf_t, 16, scale=1) / 6
hrf /= hrf.max()

subjects = []
for s in range(n_subjects):
    labels = np.tile(np.arange(n_classes), trials_per_class)
    rng.shuffle(labels)
    onsets = 15 + np.arange(n_trials) * trial_dur            # seconds

    # Subject-specific traits (inter-subject variability)
    subj_gain = 0.6 + 0.8 * rng.random()
    chan_weight = 0.3 + rng.random(n_channels)
    latency = int(round((rng.random() * 2 - 1) * fs))         # +-1 s shift
    subj_noise = 0.7 + 0.8 * rng.random()

    # Task regressor: boxcar scaled by workload, convolved with HRF
    box = np.zeros(N)
    for k in range(n_trials):
        start = int(round(onsets[k] * fs))
        jitter = 1 + 0.5 * rng.standard_normal()      # trial-to-trial variability
        box[start:start + task_dur * fs] = level_gain[labels[k]] * jitter
    resp = np.convolve(box, hrf)[:N]
    resp = np.roll(resp, latency)

    hbo = subj_gain * np.outer(resp, chan_weight)             # N x channels
    hbr = -0.3 * hbo

    # Physiological noise (shared across channels, different weights)
    cardiac = np.sin(2 * np.pi * 1.1 * t + 2 * np.pi * rng.random())
    respir = np.sin(2 * np.pi * 0.25 * t + 2 * np.pi * rng.random())
    mayer = np.sin(2 * np.pi * 0.1 * t + 2 * np.pi * rng.random())
    drift = np.cumsum(rng.standard_normal(N))
    drift = 0.02 * drift / np.abs(drift).max() * 50
    sys_noise = 0.4 * cardiac + 0.4 * respir + 0.6 * mayer

    for c in range(n_channels):
        w = 0.5 + rng.random()
        hbo[:, c] += noise_level * subj_noise * (w * sys_noise + 0.5 * rng.standard_normal(N)) + drift
        hbr[:, c] += noise_level * subj_noise * (0.5 * w * sys_noise + 0.5 * rng.standard_normal(N)) + 0.5 * drift

    # Random motion spikes
    for _ in range(6):
        p = rng.integers(fs * 5, N - fs * 5)
        c = rng.integers(n_channels)
        hbo[p:p + 3, c] += 4 * rng.standard_normal() * noise_level

    subjects.append(dict(hbo=hbo, hbr=hbr, onsets=onsets, labels=labels))

print(f"Simulated {n_subjects} subjects, {n_channels} channels, {n_trials} trials each.")

# ------------------------- 3. PREPROCESSING ------------------------------
# Zero-phase bandpass 0.01-0.2 Hz (removes heartbeat, drift)
sos = butter(3, [0.01, 0.2], btype="bandpass", fs=fs, output="sos")

raw_example = subjects[0]["hbo"][:, 0].copy()
for s in subjects:
    s["hbo"] = sosfiltfilt(sos, s["hbo"], axis=0)
    s["hbr"] = sosfiltfilt(sos, s["hbr"], axis=0)
filt_example = subjects[0]["hbo"][:, 0].copy()

# ------------------------- 4. EPOCHING + FEATURES ------------------------
n_ep = epoch_len * fs
n_base = base_len * fs
epoch_time = np.arange(n_ep) / fs

X, y, g = [], [], []
avg_resp = np.zeros((n_ep, n_classes))
cnt = np.zeros(n_classes)

for si, s in enumerate(subjects):
    for k in range(n_trials):
        o = int(round(s["onsets"][k] * fs))
        feats = []
        for D in (s["hbo"], s["hbr"]):
            ep = D[o:o + n_ep] - D[o - n_base:o].mean(axis=0)   # baseline subtraction
            mu = ep.mean(axis=0)
            pk = ep.max(axis=0)
            sl = np.array([np.polyfit(epoch_time, ep[:, c], 1)[0] for c in range(n_channels)])
            sd = ep.std(axis=0, ddof=1)
            sk = skew(ep, axis=0, bias=False)
            feats += [mu, pk, sl, sd, sk]
        X.append(np.concatenate(feats))
        y.append(s["labels"][k])
        g.append(si)

        ep_hbo = (s["hbo"][o:o + n_ep] - s["hbo"][o - n_base:o].mean(axis=0)).mean(axis=1)
        avg_resp[:, s["labels"][k]] += ep_hbo
        cnt[s["labels"][k]] += 1

X, y, g = np.array(X), np.array(y), np.array(g)
avg_resp /= cnt
print(f"Feature matrix: {X.shape[0]} trials x {X.shape[1]} features.")

# Optional per-subject normalisation (reduces inter-subject differences)
if subject_norm:
    for si in range(n_subjects):
        m = g == si
        X[m] = (X[m] - X[m].mean(axis=0)) / (X[m].std(axis=0, ddof=1) + 1e-12)

# ------------------------- 5. LOSO CROSS-VALIDATION ----------------------
models = {
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0)),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0),
}
acc = {name: np.zeros(n_subjects) for name in models}
pred = {name: np.zeros_like(y) for name in models}

for fold, (tr, te) in enumerate(LeaveOneGroupOut().split(X, y, g)):
    line = f"Held-out subject {fold + 1:2d}"
    for name, mdl in models.items():
        mdl.fit(X[tr], y[tr])
        pred[name][te] = mdl.predict(X[te])
        acc[name][fold] = np.mean(pred[name][te] == y[te])
        line += f" | {name} {acc[name][fold]:.2f}"
    print(line)

print("\nMean LOSO accuracy -> " +
      " | ".join(f"{n}: {100 * a.mean():.1f}%" for n, a in acc.items()) +
      f" | chance: {100 / n_classes:.1f}%")

# ------------------------- 6. PLOTS --------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 8))

# (a) raw vs filtered signal
w = slice(0, 120 * fs)
ax[0, 0].plot(t[w], raw_example[w], color="0.7", label="Raw (noisy)")
ax[0, 0].plot(t[w], filt_example[w], "b", lw=1.3, label="Filtered 0.01-0.2 Hz")
ax[0, 0].set(xlabel="Time (s)", ylabel="ΔHbO (a.u.)",
             title="Subject 1, channel 1: raw vs bandpass-filtered")
ax[0, 0].legend(); ax[0, 0].grid(True)

# (b) average response per class
for i, name in enumerate(class_names):
    ax[0, 1].plot(epoch_time, avg_resp[:, i], lw=2, label=name)
ax[0, 1].set(xlabel="Time after trial onset (s)", ylabel="Mean ΔHbO (a.u.)",
             title="Average HbO response by workload level")
ax[0, 1].legend(); ax[0, 1].grid(True)

# (c) accuracy per held-out subject
x = np.arange(1, n_subjects + 1)
ax[1, 0].bar(x - 0.2, 100 * acc["SVM"], 0.4, label="SVM")
ax[1, 0].bar(x + 0.2, 100 * acc["Random Forest"], 0.4, label="Random Forest")
ax[1, 0].axhline(100 / n_classes, color="r", ls="--", label="Chance")
ax[1, 0].set(xlabel="Held-out subject", ylabel="Accuracy (%)", ylim=(0, 100),
             title="Leave-One-Subject-Out accuracy")
ax[1, 0].set_xticks(x); ax[1, 0].legend(); ax[1, 0].grid(True, axis="y")

# (d) confusion matrix (SVM)
cm = confusion_matrix(y, pred["SVM"])
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax[1, 1], cmap="Blues", colorbar=False)
ax[1, 1].set_title("SVM confusion matrix (all LOSO folds)")

plt.tight_layout()
plt.savefig("brainml_demo_results.png", dpi=150)
plt.show()
