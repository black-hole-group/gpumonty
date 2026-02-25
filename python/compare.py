import numpy as np
import warnings
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import helper

# --- Load data ---
nu, nuLnu, tauabs, domega_array = helper.load_spectrum('../output/reference_sphere_thick_1e8.spec')

y_simdata = (nuLnu * domega_array[:, None] / (4 * np.pi)).sum(0)
xl_sim = np.log10(nu)
yl_sim = np.log10(y_simdata)

# Smoothed baseline
window_size = 20
y_smoothed = pd.Series(yl_sim).rolling(window=window_size, center=True).mean()

# New data
nu_new, nuLnu_new, tauabs_new, domega_array_new = helper.load_spectrum('../output/sphere_scattering_test.spec')
y_new = (nuLnu_new * domega_array_new[:, None] / (4 * np.pi)).sum(0)
xl_new = np.log10(nu_new)
yl_new = np.log10(y_new)

# --- Comparison setup ---
x, y = xl_sim, y_smoothed
xx, yy = xl_new, yl_new

LOWER_BOUND = 9
UPPER_BOUND = 21.5


def cumulative_log_difference(x, y, xx, yy, x_lower=None, x_upper=None, num_points=1000,
                              flag_threshold=0.1, excursion_threshold=0.3, verbose=True):
    overlap_min = max(x.min(), xx.min())
    overlap_max = min(x.max(), xx.max())

    x_min = x_lower if x_lower is not None else overlap_min
    x_max = x_upper if x_upper is not None else overlap_max

    if x_min >= x_max:
        raise ValueError(
            f"The lower bound ({x_min}) must be strictly less than the upper bound ({x_max})."
        )

    if x_min < overlap_min or x_max > overlap_max:
        warnings.warn(
            f"Specified bounds [{x_min}, {x_max}] exceed the overlapping domain "
            f"[{overlap_min}, {overlap_max}]. np.interp will extrapolate using flat "
            "edge values, which may affect accuracy."
        )

    x_common = np.linspace(x_min, x_max, num_points)
    y_interp = np.interp(x_common, x, y)
    yy_interp = np.interp(x_common, xx, yy)

    pointwise_log_diff = np.abs(y_interp - yy_interp)
    cum_diff_integral = cumulative_trapezoid(pointwise_log_diff, x_common, initial=0)
    x_distance = x_common - x_common[0]

    with np.errstate(divide='ignore', invalid='ignore'):
        cum_avg_log_diff = np.where(x_distance == 0, 0, cum_diff_integral / x_distance)

    total_avg_log_diff = cum_avg_log_diff[-1]
    total_linear_percent = (10**total_avg_log_diff - 1) * 100

    dex = total_avg_log_diff
    if dex < 0.01:
        label = "✅ Excellent (< 0.01 dex)"
    elif dex < 0.05:
        label = "✅ Good (0.01 – 0.05 dex)"
    elif dex < 0.1:
        label = "⚠️  Moderate (0.05 – 0.1 dex)"
    elif dex < 0.3:
        label = "🔶 Significant (0.1 – 0.3 dex)"
    else:
        label = "🔴 Severe (> 0.3 dex)"

    max_log_deviation = np.max(pointwise_log_diff)
    max_linear_factor = 10**max_log_deviation
    max_dev_index = np.argmax(pointwise_log_diff)
    max_dev_x = x_common[max_dev_index]
    has_excursion = max_log_deviation > excursion_threshold

    report = {
        "total_avg_dex": total_avg_log_diff,
        "total_avg_linear_percent": total_linear_percent,
        "label": label,
        "max_pointwise_dex": max_log_deviation,
        "max_pointwise_factor": max_linear_factor,
        "max_pointwise_x": max_dev_x,
        "has_excursion": has_excursion,
    }

    if verbose:
        print("=" * 60)
        print("  Cumulative Log-Difference Report")
        print("=" * 60)
        print(f"  Domain:             [{x_min:.6g}, {x_max:.6g}]")
        print(f"  Avg. difference:    {total_avg_log_diff:.4f} dex")
        print(f"  Avg. difference:    {total_linear_percent:.2f} %  (linear)")
        print(f"  Assessment:         {label}")
        print("-" * 60)
        print(f"  Max pointwise dev:  {max_log_deviation:.4f} dex  "
              f"(×{max_linear_factor:.2f})  at x = {max_dev_x:.6g}")
        if has_excursion:
            print(
                f"  ⚠️  WARNING: Pointwise excursion of {max_log_deviation:.4f} dex \n"
                f"      exceeds {excursion_threshold} dex threshold — localised deviation of \n"
                f"      ×{max_linear_factor:.2f} detected!"
            )
        print("=" * 60)

    return x_common, cum_avg_log_diff, total_avg_log_diff, report


# --- Run comparison ---
x_common, cum_percent_diff, log_diff, report = cumulative_log_difference(
    x, y, xx, yy,
    x_lower=LOWER_BOUND,
    x_upper=UPPER_BOUND
)

#print("\nreport =", report)

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=False)

ax1.plot(x, y, label='y(x) [Baseline]')
ax1.plot(xx, yy, label='yy(xx) [Comparison]')
ax1.axvline(LOWER_BOUND, color='gray', linestyle='--', label='Comparison Bounds')
ax1.axvline(UPPER_BOUND, color='gray', linestyle='--')
ax1.axvspan(LOWER_BOUND, UPPER_BOUND, color='gray', alpha=0.1)
ax1.set_ylabel('Function Values')
ax1.set_title('Original Functions')
ax1.set_ylim(-12, 5)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(x_common, cum_percent_diff, color='red', linewidth=2,
         label=f'Cumulative Diff (Ends at {log_diff:.2f}%)')
ax2.set_xlabel('Common x-axis')
ax2.set_ylabel('Cumulative Log10 Difference')
ax2.set_title(f'Cumulative Log Difference from x={LOWER_BOUND} to x={UPPER_BOUND}')
ax2.set_xlim(LOWER_BOUND, UPPER_BOUND)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Saves figure
OUT = '../output/sphere_scattering_comparison.png'
plt.savefig(OUT, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot saved to {OUT}")
