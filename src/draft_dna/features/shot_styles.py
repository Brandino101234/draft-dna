"""Name and visualize NMF shot styles.

Choosing k: NMF reconstruction error always falls as k grows; we pick the smallest k
after which adding a style improves the fit by less than 8% (an "elbow"), then check
the styles are interpretable. Each style is labeled by where its mass sits (rim, short
mid, long mid, corner three, above-the-break three) on the shared court grid.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

from draft_dna.features import shot_xy
from draft_dna.viz.court import draw_folded_half_court

ELBOW_GAIN = 0.08  # gains drop from ~9-12% to 5-7% after k=6 (D026)
ZONE_LABELS = {
    "rim": "Rim attacker",
    "short_mid": "Paint / floater",
    "long_mid": "Midrange",
    "corner_three": "Corner spacer",
    "above_break_three": "Above-the-break shooter",
}


def choose_k(
    maps: pd.DataFrame, ks: range = range(2, 11), seed: int = 0
) -> tuple[int, pd.DataFrame]:
    x = maps.filter(regex=r"^c\d+$").to_numpy()
    errs = []
    for k in ks:
        m = NMF(n_components=k, init="nndsvda", max_iter=1000, random_state=seed).fit(x)
        errs.append(m.reconstruction_err_)
    curve = pd.DataFrame({"k": list(ks), "error": errs})
    curve["gain"] = -curve["error"].pct_change()
    small = curve[(curve["gain"] < ELBOW_GAIN) & curve["gain"].notna()]
    k = int(small["k"].iloc[0] - 1) if len(small) else int(curve["k"].iloc[-1])
    return max(k, 2), curve


def _zone_mass(grid: np.ndarray, three_ft: float = 22.146) -> dict[str, float]:
    xx, yy = np.meshgrid(shot_xy.GRID_X, shot_xy.GRID_Y, indexing="ij")
    d = np.hypot(xx, yy)
    three = d >= three_ft - 0.5
    corner = three & (xx >= shot_xy.CORNER_DX_FT) & (yy <= shot_xy.CORNER_DY_FT)
    masks = {
        "rim": d < 4,
        "short_mid": (d >= 4) & (d < 14),
        "long_mid": (d >= 14) & ~three,
        "corner_three": corner,
        "above_break_three": three & ~corner,
    }
    total = grid.sum()
    return {z: float(grid[m].sum() / total) for z, m in masks.items()}


def describe_styles(model: NMF) -> pd.DataFrame:
    rows = []
    for i in range(model.n_components_):
        mass = _zone_mass(shot_xy.component_grid(model, i))
        top = max(mass, key=lambda z: mass[z])
        grid = shot_xy.component_grid(model, i)
        xx, yy = np.meshgrid(shot_xy.GRID_X, shot_xy.GRID_Y, indexing="ij")
        avg_ft = float((np.hypot(xx, yy) * grid).sum() / grid.sum())
        angle = float((np.degrees(np.arctan2(np.maximum(yy, 0), xx)) * grid).sum() / grid.sum())
        rows.append(
            {
                "style": f"style_{i}",
                "label": ZONE_LABELS[top],
                "avg_ft": avg_ft,
                "avg_angle": angle,
                **mass,
            }
        )
    out = pd.DataFrame(rows)
    # Disambiguate repeated labels by their second-largest zone, then by average distance.
    dup = out["label"].duplicated(keep=False)
    for i in out.index[dup]:
        zones = out.loc[i, list(ZONE_LABELS)].sort_values(ascending=False)
        out.loc[i, "label"] = (
            f"{ZONE_LABELS[zones.index[0]]} + {ZONE_LABELS[zones.index[1]].lower()}"
        )
    # Still tied: say where it comes from (baseline side vs. middle of the floor), then
    # how far out.
    still = out["label"].duplicated(keep=False)
    where = np.where(out["avg_angle"] < 45, "baseline side", "middle")
    out.loc[still, "label"] = [
        f"{lab}, {w}" for lab, w in zip(out.loc[still, "label"], where[still], strict=True)
    ]
    still = out["label"].duplicated(keep=False)
    out.loc[still, "label"] = [
        f"{lab} ({ft:.0f} ft)"
        for lab, ft in zip(out.loc[still, "label"], out.loc[still, "avg_ft"], strict=True)
    ]
    return out


def plot_styles(model: NMF, labels: pd.DataFrame, path: Path, title: str) -> None:
    k = model.n_components_
    cols = min(k, 4)
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.6 * rows), facecolor="#fcfcfb")
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        if i >= k:
            ax.axis("off")
            continue
        grid = shot_xy.component_grid(model, i)
        ax.imshow(
            grid.T,
            origin="lower",
            cmap="Blues",
            interpolation="bilinear",
            extent=(-0.5, 25.5, shot_xy.GRID_Y[0] - 0.5, shot_xy.GRID_Y[-1] + 0.5),
        )
        draw_folded_half_court(ax)
        ax.set_title(labels.loc[i, "label"], fontsize=10, color="#0b0b0b")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
