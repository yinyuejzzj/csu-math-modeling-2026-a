"""A题机械设备故障检测：论文最终可复现程序。

运行方式：在论文工作目录执行 ``python code/final_analysis.py``。
程序直接读取题目附件 data.xlsx，统一完成问题一至问题四的计算，
并在 generated 目录输出全部数值结果、数据表及论文图1—12。
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import find_peaks, spectrogram, welch
from scipy.stats import chi2, jarque_bera, ncx2


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "A题：机械设备故障检测" / "data.xlsx"
OUT = ROOT / "generated"
FIG = OUT / "figures"
TABLE = OUT / "tables"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
TABLE.mkdir(exist_ok=True)

FS = 100.0
PFA = 0.01  # 问题一全频带虚警率；问题四中表示四频率联合监测的系统总虚警率
RNG = np.random.default_rng(20260622)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_sheet(index: int):
    df = pd.read_excel(DATA_FILE, sheet_name=index)
    t = df.iloc[:, 0].to_numpy(float)
    x = df.iloc[:, 1].to_numpy(float)
    return t, x


def fft_spectrum(t, x):
    dt = float(np.median(np.diff(t)))
    y = x - np.mean(x)
    f = np.fft.rfftfreq(len(y), dt)
    p = np.abs(np.fft.rfft(y) / len(y)) ** 2
    return f, p


def ls_at_frequencies(t, x, frequencies):
    cols = []
    for f in frequencies:
        cols.extend([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)])
    phi = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(phi, x, rcond=None)
    fitted = phi @ coef
    residual = x - fitted
    params = []
    for i, f in enumerate(frequencies):
        a, b = coef[2 * i : 2 * i + 2]
        params.append(
            {
                "frequency": float(f),
                "sin_coef": float(a),
                "cos_coef": float(b),
                "amplitude": float(np.hypot(a, b)),
                "phase": float(np.arctan2(b, a)),
            }
        )
    return params, fitted, residual, float(np.sum(residual**2))


def refine_single_frequency(t, x, f_init, half_width=0.01):
    def objective(f):
        return ls_at_frequencies(t, x, [float(f)])[3]

    result = minimize_scalar(
        objective,
        bounds=(f_init - half_width, f_init + half_width),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return float(result.x)


def refine_multiple_frequencies(t, x, initial, half_width=0.01):
    initial = np.sort(np.asarray(initial, float))
    bounds = [(f - half_width, f + half_width) for f in initial]

    def objective(freqs):
        if np.any(np.diff(np.sort(freqs)) <= 0):
            return 1e12
        return ls_at_frequencies(t, x, np.sort(freqs))[3]

    result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
    return np.sort(result.x)


def q1_q2(t, x):
    f, p = fft_spectrum(t, x)
    peak = np.argmax(p[1:]) + 1
    f_fft = float(f[peak])
    f_refined = refine_single_frequency(t, x, f_fft)
    params, fitted, residual, _ = ls_at_frequencies(t, x, [f_refined])
    par = params[0]

    fw, pw = welch(
        x - x.mean(), fs=FS, window="hann", nperseg=8192, noverlap=4096
    )
    wp = np.argmax(pw[1:]) + 1

    # Conditional covariance at the refined frequency.
    phi = np.column_stack(
        [np.sin(2 * np.pi * f_refined * t), np.cos(2 * np.pi * f_refined * t)]
    )
    sigma2 = np.sum(residual**2) / (len(x) - 2)
    cov = sigma2 * np.linalg.inv(phi.T @ phi)
    draws = RNG.multivariate_normal(
        [par["sin_coef"], par["cos_coef"]], cov, size=20000
    )
    amp_draw = np.hypot(draws[:, 0], draws[:, 1])
    phase_draw = np.arctan2(draws[:, 1], draws[:, 0])

    # Residual diagnostics and split-sample stability.
    residual_centered = residual - residual.mean()
    residual_var = float(np.dot(residual_centered, residual_centered))
    residual_acf = [
        float(np.dot(residual_centered[:-lag], residual_centered[lag:]) / residual_var)
        for lag in range(1, 21)
    ]
    jb = jarque_bera(residual_centered)
    half = len(x) // 2
    first_params, *_ = ls_at_frequencies(t[:half], x[:half], [f_refined])
    second_params, *_ = ls_at_frequencies(t[half:], x[half:], [f_refined])

    # Global FFT maximum threshold over 0--20 Hz under exponential-bin approximation.
    bins = int(np.sum((f > 0) & (f <= 20)))
    global_threshold_over_mean = -np.log(1 - (1 - PFA) ** (1 / bins))
    observed_over_mean = float(p[peak] / np.mean(p[(f > 0) & (f <= 20)]))

    # Cross-segment evidence and observed noise-floor fluctuation for the
    # comparison with a single traditional periodogram.  The eight Hann
    # segments are exactly those averaged by the Welch estimate.
    fseg, tseg, pseg = spectrogram(
        x - x.mean(), fs=FS, window="hann", nperseg=8192, noverlap=4096,
        mode="psd"
    )
    search_seg = (fseg > 0) & (fseg <= 20)
    segment_peaks = fseg[search_seg][np.argmax(pseg[search_seg, :], axis=0)]
    segment_detection_rate = float(
        np.mean(np.abs(segment_peaks - f_refined) <= FS / 8192)
    )
    noise_fft = (f > 0) & (f <= 20) & (np.abs(f - f_refined) > 0.2)
    noise_welch = (fw > 0) & (fw <= 20) & (np.abs(fw - f_refined) > 0.2)
    fft_noise_cv = float(np.std(p[noise_fft]) / np.mean(p[noise_fft]))
    welch_noise_cv = float(np.std(pw[noise_welch]) / np.mean(pw[noise_welch]))

    result = {
        "N": len(x),
        "fs": float(1 / np.median(np.diff(t))),
        "duration": float(t[-1] - t[0]),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "f_fft_bin": f_fft,
        "f_refined": f_refined,
        "fft_peak_over_median": float(p[peak] / np.median(p[1:])),
        "welch_L": 8192,
        "welch_peak": float(fw[wp]),
        "welch_peak_over_median": float(pw[wp] / np.median(pw[1:])),
        "global_bins_0_20": bins,
        "global_threshold_over_mean_pfa_0_01": float(global_threshold_over_mean),
        "observed_peak_over_mean_0_20": observed_over_mean,
        "welch_validation_segments": int(pseg.shape[1]),
        "segment_detection_rate": segment_detection_rate,
        "fft_noise_floor_cv": fft_noise_cv,
        "welch_noise_floor_cv": welch_noise_cv,
        "amplitude": par["amplitude"],
        "phase": par["phase"],
        "rmse_residual": float(np.sqrt(np.mean(residual**2))),
        "amplitude_ci95_conditional": np.quantile(amp_draw, [0.025, 0.975]).tolist(),
        "phase_ci95_conditional": np.quantile(phase_draw, [0.025, 0.975]).tolist(),
        "residual_max_abs_acf_lag_1_20": float(np.max(np.abs(residual_acf))),
        "residual_jarque_bera_stat": float(jb.statistic),
        "residual_jarque_bera_p": float(jb.pvalue),
        "split_first_amplitude": first_params[0]["amplitude"],
        "split_second_amplitude": second_params[0]["amplitude"],
        "split_first_phase": first_params[0]["phase"],
        "split_second_phase": second_params[0]["phase"],
    }

    # Figure 1: data evidence for Q1.
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    mask_t = t <= 5
    ax[0, 0].plot(t[mask_t], x[mask_t], lw=0.65)
    ax[0, 0].set(title="(a) 单源信号前5 s", xlabel="时间 / s", ylabel="幅值 / a.u.")
    m = (f > 0) & (f <= 20)
    ax[0, 1].semilogy(f[m], p[m], lw=0.7)
    ax[0, 1].axvline(f_refined, color="r", ls="--", label=f"f={f_refined:.6f} Hz")
    ax[0, 1].set(title="(b) 全段FFT功率谱", xlabel="频率 / Hz", ylabel="功率")
    ax[0, 1].legend()
    mw = (fw > 0) & (fw <= 20)
    ax[1, 0].semilogy(fw[mw], pw[mw], color="tab:green", lw=0.8)
    ax[1, 0].axvline(fw[wp], color="r", ls="--", label=f"f={fw[wp]:.6f} Hz")
    ax[1, 0].set(title="(c) Welch功率谱（L=8192）", xlabel="频率 / Hz", ylabel="PSD")
    ax[1, 0].legend()
    zf = (f >= 1.8) & (f <= 2.2)
    zw = (fw >= 1.8) & (fw <= 2.2)
    ax[1, 1].plot(f[zf], p[zf] / p[zf].max(), label="FFT")
    ax[1, 1].plot(fw[zw], pw[zw] / pw[zw].max(), label="Welch")
    ax[1, 1].axvline(f_refined, color="r", ls="--")
    ax[1, 1].set(title="(d) 2 Hz附近归一化谱", xlabel="频率 / Hz", ylabel="归一化功率")
    ax[1, 1].legend()
    for a in ax.ravel():
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "图1.png", dpi=220)
    plt.close(fig)

    # Figure 2: direct, data-based comparison; no incompatible P_D values are mixed.
    fstft, tstft, pstft = spectrogram(
        x - x.mean(), fs=FS, window="hann", nperseg=256, noverlap=128, mode="psd"
    )
    pstft_mean = pstft.mean(axis=1)
    stft_peak = int(np.argmax(pstft_mean[1:]) + 1)
    rows = [
        {
            "method": "STFT",
            "L": 256,
            "delta_f": FS / 256,
            "segments": int(len(tstft)),
            "peak_over_median": float(pstft_mean[stft_peak] / np.median(pstft_mean[1:])),
            "noise_floor_cv": np.nan,
            "segment_detection_rate": np.nan,
            "global_pfa_control": "否",
        },
        {
            "method": "全段FFT",
            "L": len(x),
            "delta_f": FS / len(x),
            "segments": 1,
            "peak_over_median": float(p[peak] / np.median(p[1:])),
            "noise_floor_cv": fft_noise_cv,
            "segment_detection_rate": np.nan,
            "global_pfa_control": "否",
        },
        {
            "method": "Welch",
            "L": 8192,
            "delta_f": FS / 8192,
            "segments": 8,
            "peak_over_median": float(pw[wp] / np.median(pw[1:])),
            "noise_floor_cv": welch_noise_cv,
            "segment_detection_rate": segment_detection_rate,
            "global_pfa_control": "与最大谱线检验联合后为是",
        },
    ]
    pd.DataFrame(rows).to_csv(TABLE / "q1_method_comparison.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].bar([r["method"] for r in rows], [r["delta_f"] for r in rows])
    ax[0].set_yscale("log")
    ax[0].set(title="频率分辨率", ylabel=r"$\Delta f$ / Hz")
    ax[1].bar([r["method"] for r in rows], [r["peak_over_median"] for r in rows])
    ax[1].set_yscale("log")
    ax[1].set(title="实测峰值/谱中位基底", ylabel="比值")
    for a in ax:
        a.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "图2.png", dpi=220)
    plt.close(fig)

    # Q2 figures.
    fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    mask = t <= 3
    ax[0].plot(t[mask], x[mask], lw=0.6, alpha=0.7, label="观测信号")
    ax[0].plot(t[mask], fitted[mask], color="r", lw=1.6, label="恢复周期分量")
    ax[0].legend()
    ax[0].set(title="观测信号与恢复周期分量", ylabel="幅值 / a.u.")
    ax[1].plot(t[mask], fitted[mask], color="r")
    ax[1].set(title="恢复周期分量", ylabel="幅值 / a.u.")
    ax[2].plot(t[mask], residual[mask], color="0.45", lw=0.65)
    ax[2].set(title=f"残差（RMSE={result['rmse_residual']:.4f}）", xlabel="时间 / s", ylabel="幅值 / a.u.")
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "图3.png", dpi=220)
    plt.close(fig)

    fr, pr = fft_spectrum(t, residual)
    ff, pf = fft_spectrum(t, fitted)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    m = (f > 0) & (f <= 16)
    ax[0].semilogy(f[m], p[m], label="观测")
    ax[0].semilogy(ff[m], pf[m], label="恢复")
    ax[0].semilogy(fr[m], pr[m], label="残差", alpha=0.8)
    ax[0].set(title="全频段对比", xlabel="频率 / Hz", ylabel="功率")
    mz = (f >= 1.8) & (f <= 2.2)
    ax[1].semilogy(f[mz], p[mz], label="观测")
    ax[1].semilogy(ff[mz], pf[mz], label="恢复")
    ax[1].semilogy(fr[mz], pr[mz], label="残差")
    ax[1].set(title="2 Hz附近", xlabel="频率 / Hz", ylabel="功率")
    for a in ax:
        a.grid(alpha=0.25)
        a.legend()
    fig.tight_layout()
    fig.savefig(FIG / "图4.png", dpi=220)
    plt.close(fig)

    # Q2 controlled truth experiment.  The clean sinusoid is known, while
    # noise is sampled from the centered attachment residual so that its
    # scale and non-parametric marginal distribution match the observed data.
    sim_rng = np.random.default_rng(20260623)
    n_sim = 500
    clean = par["amplitude"] * np.sin(2 * np.pi * f_refined * t + par["phase"])
    clean_rms = float(np.sqrt(np.mean(clean**2)))
    sim_rows = []
    for _ in range(n_sim):
        y_sim = clean + sim_rng.choice(residual_centered, size=len(t), replace=True)
        fsim, psim = fft_spectrum(t, y_sim)
        local = (fsim >= f_refined - 0.02) & (fsim <= f_refined + 0.02)
        local_ids = np.flatnonzero(local)
        k = int(local_ids[np.argmax(psim[local])])
        # Three-bin parabolic interpolation of the FFT peak gives a fast
        # continuous-frequency refinement for the 500-run validation.
        if 0 < k < len(psim) - 1:
            denom = psim[k - 1] - 2 * psim[k] + psim[k + 1]
            delta = 0.5 * (psim[k - 1] - psim[k + 1]) / denom if denom != 0 else 0.0
            delta = float(np.clip(delta, -0.5, 0.5))
        else:
            delta = 0.0
        fhat_sim = float(fsim[k] + delta * (fsim[1] - fsim[0]))
        p_sim, fit_sim, _, _ = ls_at_frequencies(t, y_sim, [fhat_sim])
        amp_error = abs(p_sim[0]["amplitude"] - par["amplitude"]) / par["amplitude"]
        phase_delta = np.angle(np.exp(1j * (p_sim[0]["phase"] - par["phase"])))
        sim_rows.append(
            {
                "frequency_abs_error_hz": abs(fhat_sim - f_refined),
                "amplitude_relative_error": amp_error,
                "phase_abs_error_rad": abs(phase_delta),
                "waveform_nrmse": float(np.sqrt(np.mean((fit_sim - clean) ** 2)) / clean_rms),
            }
        )
    sim_df = pd.DataFrame(sim_rows)
    sim_df.to_csv(TABLE / "q2_truth_simulation.csv", index=False)
    sim_summary = {
        "n": n_sim,
        "noise_source": "centered attachment residual resampling",
    }
    for col in sim_df.columns:
        sim_summary[col] = {
            "median": float(sim_df[col].median()),
            "mean": float(sim_df[col].mean()),
            "q025": float(sim_df[col].quantile(0.025)),
            "q975": float(sim_df[col].quantile(0.975)),
        }
    result["truth_simulation"] = sim_summary
    return result


def q3(t, x):
    f, p = fft_spectrum(t, x)
    noise_median = np.median(p[1:])
    peaks, props = find_peaks(p, height=5 * noise_median, distance=2)
    peaks = peaks[np.argsort(p[peaks])[::-1]]
    top = peaks[:10]

    bic_rows = []
    fixed_models = {}
    for m in range(1, 11):
        freqs = np.sort(f[top[:m]])
        params, fitted, residual, rss = ls_at_frequencies(t, x, freqs)
        # Conservatively count frequency, sine and cosine parameters: k_M=3M.
        bic = len(x) * np.log(rss / len(x)) + 3 * m * np.log(len(x))
        bic_rows.append({"M": m, "RSS": rss, "BIC": bic})
        fixed_models[m] = (freqs, params, fitted, residual)
    m_opt = int(min(bic_rows, key=lambda r: r["BIC"])["M"])
    initial = fixed_models[m_opt][0]
    refined = refine_multiple_frequencies(t, x, initial, half_width=0.005)
    params, fitted, residual, rss = ls_at_frequencies(t, x, refined)
    params = sorted(params, key=lambda z: z["frequency"])

    # Conditional residual bootstrap for BIC order-selection stability.
    # Candidate frequencies remain fixed; the experiment tests stability to residual noise.
    n_boot = 500
    model_q = []
    for m in range(1, 11):
        freqs_m = np.sort(f[top[:m]])
        cols = []
        for fm in freqs_m:
            cols.extend([np.sin(2 * np.pi * fm * t), np.cos(2 * np.pi * fm * t)])
        phi_m = np.column_stack(cols)
        q_m, _ = np.linalg.qr(phi_m, mode="reduced")
        model_q.append(q_m)
    base_fit = fixed_models[m_opt][2]
    base_residual = fixed_models[m_opt][3] - np.mean(fixed_models[m_opt][3])
    selected_orders = []
    batch_size = 20
    for start in range(0, n_boot, batch_size):
        b = min(batch_size, n_boot - start)
        ids = RNG.integers(0, len(x), size=(len(x), b))
        yb = base_fit[:, None] + base_residual[ids]
        y_energy = np.sum(yb**2, axis=0)
        scores = []
        for m, q_m in enumerate(model_q, 1):
            projection = q_m.T @ yb
            rss_b = np.maximum(y_energy - np.sum(projection**2, axis=0), 1e-15)
            scores.append(len(x) * np.log(rss_b / len(x)) + 3 * m * np.log(len(x)))
        selected_orders.extend((np.argmin(np.vstack(scores), axis=0) + 1).tolist())
    bootstrap_counts = {m: int(selected_orders.count(m)) for m in range(1, 11)}
    pd.DataFrame(
        [{"M": m, "count": bootstrap_counts[m], "proportion": bootstrap_counts[m] / n_boot}
         for m in range(1, 11)]
    ).to_csv(TABLE / "q3_bootstrap_order.csv", index=False)

    pd.DataFrame(bic_rows).to_csv(TABLE / "q3_bic.csv", index=False)
    pd.DataFrame(params).to_csv(TABLE / "q3_parameters.csv", index=False)

    # CRLB with unknown phase: centered-time Fisher information approximation.
    sigma = float(np.std(residual))
    a_ref = 0.025
    duration = float(t[-1] - t[0])
    crlb_std_hz = np.sqrt(
        24 * sigma**2 / (a_ref**2 * (2 * np.pi) ** 2 * len(x) * duration**2)
    )

    # Empirical two-peak resolution experiment.
    deltas = np.array([0.0025, 0.005, 0.0075, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    n_mc = 500
    resolution_rows = []
    n = len(t)
    df = FS / n
    local_freq = np.fft.rfftfreq(n, 1 / FS)
    local_mask = (local_freq >= 12.7) & (local_freq <= 13.8)
    local_ids = np.flatnonzero(local_mask)
    for delta in deltas:
        success = 0
        amp_errors = []
        for _ in range(n_mc):
            ph1, ph2 = RNG.uniform(-np.pi, np.pi, 2)
            y = (
                0.025 * np.sin(2 * np.pi * 13.0 * t + ph1)
                + 0.010 * np.sin(2 * np.pi * (13.0 + delta) * t + ph2)
                + RNG.normal(0, sigma, n)
            )
            spec = np.abs(np.fft.rfft(y) / n) ** 2
            lp, _ = find_peaks(spec[local_ids], distance=1)
            if len(lp) < 2:
                continue
            chosen = local_ids[lp[np.argsort(spec[local_ids[lp]])[-2:]]]
            est = np.sort(local_freq[chosen])
            tol = max(2.2 * df, 0.25 * delta)
            if abs(est[0] - 13.0) <= tol and abs(est[1] - (13.0 + delta)) <= tol:
                pp, *_ = ls_at_frequencies(t, y, est)
                aa = np.array(sorted([v["amplitude"] for v in pp], reverse=True))
                err = np.mean(np.abs(aa - np.array([0.025, 0.010])) / np.array([0.025, 0.010]))
                amp_errors.append(err)
                success += 1
        phat = success / n_mc
        z = 1.959963984540054
        denom = 1.0 + z**2 / n_mc
        center = (phat + z**2 / (2 * n_mc)) / denom
        half = z * np.sqrt(phat * (1 - phat) / n_mc + z**2 / (4 * n_mc**2)) / denom
        resolution_rows.append(
            {
                "delta_f": float(delta),
                "success_rate": phat,
                "success_wilson95_low": float(center - half),
                "success_wilson95_high": float(center + half),
                "mean_amplitude_relative_error": float(np.mean(amp_errors)) if amp_errors else np.nan,
                "n_mc": n_mc,
            }
        )
    pd.DataFrame(resolution_rows).to_csv(TABLE / "q3_resolution.csv", index=False)

    # Figures 5--7.
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    m = (f > 0) & (f <= 25)
    ax[0].semilogy(f[m], p[m], lw=0.65)
    for idx, par in enumerate(params):
        ax[0].axvline(par["frequency"], color="r", ls="--", alpha=0.7)
        ax[0].text(
            par["frequency"],
            0.96 - 0.14 * (idx % 2),
            f"{par['frequency']:.4f} Hz",
            rotation=90,
            ha="center",
            va="top",
            transform=ax[0].get_xaxis_transform(),
            fontsize=9,
        )
    ax[0].set(title="多源FFT候选峰", xlabel="频率 / Hz", ylabel="功率")
    ax[1].plot([r["M"] for r in bic_rows], [r["BIC"] for r in bic_rows], marker="o")
    ax[1].axvline(m_opt, color="r", ls="--", label=f"M*={m_opt}")
    ax[1].set(title="BIC模型阶数选择", xlabel="模型阶数 M", ylabel="BIC")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "图5.png", dpi=220)
    plt.close(fig)

    mask = t <= 3
    fig, ax = plt.subplots(len(params) + 1, 1, figsize=(14, 12), sharex=True)
    components = []
    for i, par in enumerate(params):
        comp = par["amplitude"] * np.sin(2 * np.pi * par["frequency"] * t + par["phase"])
        components.append(comp)
        ax[i].plot(t[mask], x[mask], color="0.8", lw=0.5)
        ax[i].plot(t[mask], comp[mask], color="r", lw=1.2)
        ax[i].set_ylabel(f"{par['frequency']:.3f} Hz")
        ax[i].grid(alpha=0.2)
    ax[-1].plot(t[mask], x[mask], color="0.75", lw=0.5, label="观测")
    ax[-1].plot(t[mask], np.sum(components, axis=0)[mask], color="r", lw=1.1, label="联合恢复")
    ax[-1].set(xlabel="时间 / s", ylabel="总和")
    ax[-1].legend()
    ax[-1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "图6.png", dpi=220)
    plt.close(fig)

    rr = pd.DataFrame(resolution_rows)
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.semilogx(rr.delta_f, rr.success_rate, marker="o", color="tab:blue")
    ax1.fill_between(
        rr.delta_f,
        rr.success_wilson95_low,
        rr.success_wilson95_high,
        color="tab:blue",
        alpha=0.18,
        label="Wilson 95%区间",
    )
    ax1.axhline(0.95, color="r", ls="--", label="95%可靠分离标准")
    ax1.set(xlabel=r"频率间距 $\Delta f$ / Hz", ylabel="双峰识别成功率", ylim=(0, 1.05))
    ax2 = ax1.twinx()
    ax2.semilogx(rr.delta_f, 100 * rr.mean_amplitude_relative_error, marker="s", color="tab:orange")
    ax2.set_ylabel("成功试验中的平均振幅相对误差 / %")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "图7.png", dpi=220)
    plt.close(fig)

    return {
        "N": len(x),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "candidate_count_5x_median": int(len(peaks)),
        "M_opt": m_opt,
        "delta_BIC_5_minus_4": float(bic_rows[4]["BIC"] - bic_rows[3]["BIC"]),
        "bootstrap_order_n": n_boot,
        "bootstrap_order_counts": bootstrap_counts,
        "bootstrap_order_m4_proportion": bootstrap_counts[4] / n_boot,
        "parameters": params,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "crlb_frequency_std_unknown_phase_hz": float(crlb_std_hz),
        "resolution": resolution_rows,
    }


def q4(q3_result, sigma_base):
    # The geometry below is an explicit illustrative scenario, not supplied by the problem.
    width, height = 4.0, 2.0
    source_positions = np.array([[0.5, 1.0], [1.5, 0.3], [2.5, 1.7], [3.5, 1.0]])
    source_frequencies = np.array([4.0, 8.0, 13.0, 14.0])
    q3_map = {round(p["frequency"]): p["amplitude"] for p in q3_result["parameters"]}
    source_amplitudes = np.array([q3_map[int(f)] for f in source_frequencies])

    xs = np.arange(0.25, width, 0.25)
    ys = np.arange(0.25, height, 0.25)
    candidates = np.array([(x, y) for x in xs for y in ys])
    assert len(candidates) == 105

    alpha = 0.5
    t_obs = 4.0
    n_obs = int(FS * t_obs)
    n_monitored = len(source_positions)

    def per_source_pfa(system_pfa):
        """Bonferroni allocation, valid without assuming independent frequency tests."""
        return system_pfa / n_monitored

    def noise_sigma(points, scale=1.0):
        # Representative non-uniform motor-noise field, strongest near the device center.
        r2 = (points[:, 0] - 2.0) ** 2 + (points[:, 1] - 1.0) ** 2
        return sigma_base * scale * (1.0 + 0.35 * np.exp(-r2 / (2 * 0.6**2)))

    def rho_matrix(alpha_value=alpha, noise_scale=1.0, amplitudes=source_amplitudes):
        d = np.linalg.norm(candidates[:, None, :] - source_positions[None, :, :], axis=2)
        sig = noise_sigma(candidates, noise_scale)[:, None]
        return (amplitudes[None, :] ** 2 * np.exp(-2 * alpha_value * d)) / (2 * sig**2)

    def pd_from_lambda(lam, q, system_pfa=PFA):
        threshold = chi2.ppf(1 - per_source_pfa(system_pfa), 2 * q)
        return ncx2.sf(threshold, 2 * q, lam)

    def optimize_q(q, rho):
        best_score = -1.0
        best_ids = None
        best_rho = None
        for ids in itertools.combinations(range(len(candidates)), q):
            combined = rho[list(ids)].sum(axis=0)
            score = float(combined.min())
            if score > best_score:
                best_score, best_ids, best_rho = score, ids, combined
        lam = n_obs * best_rho
        pds = pd_from_lambda(lam, q)
        return {
            "q": q,
            "ids": list(best_ids),
            "positions": candidates[list(best_ids)].tolist(),
            "rho_by_source": best_rho.tolist(),
            "rho_min": float(best_rho.min()),
            "snr_min_db": float(10 * np.log10(best_rho.min())),
            "pd_by_source": pds.tolist(),
            "pd_min": float(pds.min()),
        }

    def evaluate_ids(ids, q, rho):
        combined = rho[list(ids)].sum(axis=0)
        pds = pd_from_lambda(n_obs * combined, q)
        return {
            "q": q,
            "ids": list(ids),
            "positions": candidates[list(ids)].tolist(),
            "rho_by_source": combined.tolist(),
            "rho_min": float(combined.min()),
            "snr_min_db": float(10 * np.log10(combined.min())),
            "pd_by_source": pds.tolist(),
            "pd_min": float(pds.min()),
        }

    rho = rho_matrix()
    layouts_scenario_specific = [optimize_q(q, rho) for q in [1, 2, 3]]

    # The problem does not map the four detected amplitudes to four physical positions.
    # Design conservatively for the weakest amplitude, strongest attenuation/noise,
    # and the lower sensitivity bound, then evaluate under the nominal scenario.
    weakest = float(source_amplitudes.min())
    gain_min = 0.85
    robust_rho = gain_min**2 * rho_matrix(
        alpha_value=0.8,
        noise_scale=1.2,
        amplitudes=np.full(4, weakest),
    )
    robust_designs = [optimize_q(q, robust_rho) for q in [1, 2, 3]]
    layouts = [evaluate_ids(d["ids"], d["q"], rho) for d in robust_designs]
    best = layouts[-1]

    # Robustness over attenuation, noise scale and all frequency-position assignments.
    sensitivity_rows = []
    for a in [0.3, 0.4, 0.5, 0.6, 0.8]:
        for ns in [0.8, 1.0, 1.2]:
            rr = rho_matrix(
                alpha_value=a,
                noise_scale=ns,
                amplitudes=np.full(4, weakest),
            )
            # Evaluate the same robust layout; do not re-optimize after perturbation.
            sol = evaluate_ids(best["ids"], 3, rho_matrix(alpha_value=a, noise_scale=ns))
            sensitivity_rows.append(
                {
                    "alpha": a,
                    "noise_scale": ns,
                    "positions": json.dumps(sol["positions"], ensure_ascii=False),
                    "snr_min_db": sol["snr_min_db"],
                    "pd_min": sol["pd_min"],
                }
            )
    pd.DataFrame(sensitivity_rows).to_csv(TABLE / "q4_sensitivity.csv", index=False)

    # Worst amplitude-to-position permutation evaluated for the baseline layout.
    perm_pd = []
    for amps in itertools.permutations(source_amplitudes):
        r = rho_matrix(amplitudes=np.asarray(amps))[best["ids"]].sum(axis=0)
        perm_pd.append(float(pd_from_lambda(n_obs * r, 3).min()))

    # Monte Carlo calibration at orthogonal matched-filter-output level.
    # Four simultaneous detectors are simulated to verify family-wise P_FA.
    pfa_single = per_source_pfa(PFA)
    threshold = chi2.ppf(1 - pfa_single, 6)
    mc_n = 100000
    h0_stats = RNG.chisquare(6, size=(mc_n, n_monitored))
    empirical_pfa = float(np.mean(np.any(h0_stats > threshold, axis=1)))
    empirical_pd = []
    for lam in n_obs * np.asarray(best["rho_by_source"]):
        phase = RNG.uniform(-np.pi, np.pi, size=(mc_n, 3))
        lam_sensor = np.full(3, lam / 3.0)
        means = np.stack(
            [np.sqrt(lam_sensor)[None, :] * np.cos(phase),
             np.sqrt(lam_sensor)[None, :] * np.sin(phase)],
            axis=2,
        )
        z = RNG.normal(size=(mc_n, 3, 2)) + means
        empirical_pd.append(float(np.mean(np.sum(z**2, axis=(1, 2)) > threshold)))

    # Measurement-position sensitivity is not supplied by the problem. Keep g_ij=1
    # in the baseline and quantify ±15% multiplicative uncertainty for the selected layout.
    selected_base_rho = rho[best["ids"]]
    gain_pd = []
    for _ in range(5000):
        gains = RNG.uniform(0.85, 1.15, size=selected_base_rho.shape)
        combined = (selected_base_rho * gains**2).sum(axis=0)
        gain_pd.append(float(pd_from_lambda(n_obs * combined, 3).min()))

    # Figure 8: scenario and optimized three-sensor layout.
    gx, gy = np.meshgrid(np.linspace(0, width, 161), np.linspace(0, height, 81))
    gp = np.column_stack([gx.ravel(), gy.ravel()])
    gd = np.linalg.norm(gp[:, None, :] - source_positions[None, :, :], axis=2)
    gs = noise_sigma(gp)[:, None]
    field = np.sum(source_amplitudes[None, :] ** 2 * np.exp(-2 * alpha * gd) / (2 * gs**2), axis=1)
    field = 10 * np.log10(field.reshape(gx.shape))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    c = ax.contourf(gx, gy, field, levels=30, cmap="YlOrRd")
    fig.colorbar(c, ax=ax, label="四源累积线性SNR / dB")
    ax.scatter(source_positions[:, 0], source_positions[:, 1], marker="*", s=230, c="red", edgecolor="k", label="假设故障源")
    bp = np.asarray(best["positions"])
    ax.scatter(bp[:, 0], bp[:, 1], marker="^", s=180, c="cyan", edgecolor="k", label="最优传感器")
    for i, (xy, f0, aa) in enumerate(zip(source_positions, source_frequencies, source_amplitudes), 1):
        ax.text(xy[0] + 0.04, xy[1] + 0.06, f"S{i}: {f0:.0f} Hz, A={aa:.3f}")
    ax.set(xlim=(0, width), ylim=(0, height), xlabel="x / m", ylabel="y / m", title="代表性4 m×2 m场景中的三传感器布局")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "图8.png", dpi=220)
    plt.close(fig)

    # Figure 9: ROC and amplitude-scale performance.
    pfas = np.logspace(-5, 0, 180)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for sol in layouts:
        r = np.asarray(sol["rho_by_source"])
        curve = [float(pd_from_lambda(n_obs * r, sol["q"], p).min()) for p in pfas]
        ax[0].semilogx(pfas, curve, label=f"{sol['q']}个传感器")
    ax[0].axvline(PFA, color="k", ls="--")
    ax[0].set(xlabel=r"系统总 $P_{FA}^{sys}$", ylabel=r"系统最差 $P_D$", title="ROC曲线", ylim=(0, 1.02))
    scales_db = np.linspace(-12, 12, 100)
    for sol in layouts:
        r = np.asarray(sol["rho_by_source"])
        curve = [float(pd_from_lambda(n_obs * r * 10 ** (db / 10), sol["q"]).min()) for db in scales_db]
        ax[1].plot(scales_db, curve, label=f"{sol['q']}个传感器")
    ax[1].axvline(0, color="k", ls="--", label="问题三估计振幅")
    ax[1].set(xlabel="故障功率尺度变化 / dB", ylabel=r"系统最差 $P_D$", title=r"固定系统总 $P_{FA}=0.01$", ylim=(0, 1.02))
    for a in ax:
        a.grid(alpha=0.25)
        a.legend()
    fig.tight_layout()
    fig.savefig(FIG / "图9.png", dpi=220)
    plt.close(fig)

    # Figure 10: independent numerical calibration of theory at detector-output level.
    theory_pd = np.asarray(best["pd_by_source"])
    xbar = np.arange(n_monitored)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(xbar - 0.18, theory_pd, 0.36, label="非中心卡方理论值")
    ax.bar(xbar + 0.18, empirical_pd, 0.36, label="蒙特卡洛值")
    ax.set_xticks(xbar, [f"源{i}" for i in range(1, n_monitored + 1)])
    ax.set(
        ylabel=r"$P_D$",
        ylim=(0, 1.02),
        title=rf"理论与蒙特卡洛校准（系统 $P_{{FA}}$={PFA:.2f}，经验值={empirical_pfa:.4f}）",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "图10.png", dpi=220)
    plt.close(fig)

    # Figure 11: observation-time effect with a consistent sample count.
    times = np.unique(np.r_[np.logspace(0, 3, 80), 4, 400])
    fig, ax = plt.subplots(figsize=(9, 5))
    for sol in layouts:
        r = np.asarray(sol["rho_by_source"])
        curve = [float(pd_from_lambda(int(round(FS * tt)) * r, sol["q"]).min()) for tt in times]
        ax.semilogx(times, curve, label=f"{sol['q']}个传感器")
    ax.axvline(4, color="k", ls="--", label="4 s实时场景")
    ax.axvline(400, color="0.5", ls=":", label="400 s完整数据")
    ax.set(xlabel="观测时长 / s", ylabel=r"系统最差 $P_D$", ylim=(0, 1.02), title=r"固定 $P_{FA}=0.01$")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "图11.png", dpi=220)
    plt.close(fig)

    time_to_target = {}
    for sol in layouts:
        r = np.asarray(sol["rho_by_source"])
        time_to_target[str(sol["q"])] = {}
        for target in [0.5, 0.9]:
            lo, hi = 0.01, 5000.0
            for _ in range(70):
                mid = 0.5 * (lo + hi)
                value = float(pd_from_lambda(max(1, int(round(FS * mid))) * r, sol["q"]).min())
                if value >= target:
                    hi = mid
                else:
                    lo = mid
            time_to_target[str(sol["q"])][str(target)] = hi

    # Figure 12: false-alarm sensitivity, generated from the same layouts.
    pfa_levels = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
    xloc = np.arange(len(pfa_levels))
    fig, ax = plt.subplots(figsize=(10, 5))
    width_bar = 0.23
    pfa_rows = []
    for k, sol in enumerate(layouts):
        r = np.asarray(sol["rho_by_source"])
        vals = [float(pd_from_lambda(n_obs * r, sol["q"], p).min()) for p in pfa_levels]
        ax.bar(xloc + (k - 1) * width_bar, vals, width_bar, label=f"{sol['q']}个传感器")
        for p, v in zip(pfa_levels, vals):
            pfa_rows.append({"q": sol["q"], "P_FA": p, "P_D_min": v})
    ax.set_xticks(xloc, [str(v) for v in pfa_levels])
    ax.set(xlabel=r"系统总 $P_{FA}^{sys}$", ylabel=r"系统最差 $P_D$", ylim=(0, 1.02), title="系统虚警率约束敏感性")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "图12.png", dpi=220)
    plt.close(fig)
    pd.DataFrame(pfa_rows).to_csv(TABLE / "q4_pfa.csv", index=False)
    pd.DataFrame(layouts).to_json(TABLE / "q4_layouts.json", orient="records", force_ascii=False, indent=2)

    return {
        "scenario_notice": "4m×2m geometry, source locations and spatial noise field are illustrative assumptions, not problem-given data.",
        "width": width,
        "height": height,
        "source_positions": source_positions.tolist(),
        "source_frequencies": source_frequencies.tolist(),
        "source_amplitudes_from_q3": source_amplitudes.tolist(),
        "candidate_count": len(candidates),
        "combination_count_q3": int(len(list(itertools.combinations(range(len(candidates)), 3)))),
        "alpha": alpha,
        "noise_model": "sigma(p)=sigma0*[1+0.35 exp(-||p-(2,1)||^2/(2*0.6^2))]",
        "sigma0": sigma_base,
        "observation_time_primary_s": t_obs,
        "system_pfa": PFA,
        "per_source_pfa_bonferroni": float(pfa_single),
        "per_source_pfa_sidak_independence_reference": float(1.0 - (1.0 - PFA) ** (1.0 / n_monitored)),
        "layouts": layouts,
        "layouts_scenario_specific": layouts_scenario_specific,
        "layout_policy": "robust design uses weakest amplitude, alpha=0.8, noise scale=1.2 and g_min=0.85 at every potential source position",
        "permutation_worst_pd_for_baseline_q3_layout": float(min(perm_pd)),
        "permutation_best_pd_for_baseline_q3_layout": float(max(perm_pd)),
        "empirical_pfa": empirical_pfa,
        "empirical_pd_by_source": empirical_pd,
        "time_to_target_pd_s": time_to_target,
        "measurement_sensitivity_model": "baseline g_ij=1; robustness test g_ij independently varies in [0.85,1.15]",
        "measurement_sensitivity_pd_quantiles_5_50_95": np.quantile(gain_pd, [0.05, 0.5, 0.95]).tolist(),
    }


def main():
    t1, x1 = load_sheet(0)
    t3, x3 = load_sheet(1)
    q12 = q1_q2(t1, x1)
    q3_result = q3(t3, x3)
    q4_result = q4(q3_result, q12["rmse_residual"])
    results = {"q1_q2": q12, "q3": q3_result, "q4": q4_result}
    with open(OUT / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
