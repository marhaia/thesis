"""Pinned windowed (local) SSIM — the single implementation used by the resize
causality comparator, reported separately from the previous audit's global SSIM.

This is a self-contained Wang et al. (2004) mean-window SSIM with a fixed 7x7
uniform window and 'reflect' boundary handling, so results are fully
reproducible with numpy + scipy only (no scikit-image version dependence).

Every parameter is pinned and exported in WINDOWED_SSIM_PARAMS so the manifest
and JSON record exactly how it was computed:
    win_size   = 7            (7x7 uniform averaging window)
    K1, K2     = 0.01, 0.03
    data_range = 1.0          (inputs are min-max normalised to [0, 1])
    boundary   = 'reflect'
    C1 = (K1*data_range)^2,  C2 = (K2*data_range)^2

numpy + scipy are imported lazily (inside windowed_ssim) so this module — and
the Phase 0 manifest that imports WINDOWED_SSIM_PARAMS — loads with only the
standard library present.
"""

WINDOWED_SSIM_PARAMS = {
    "implementation": "wang2004_uniform_window",
    "win_size": 7,
    "K1": 0.01,
    "K2": 0.03,
    "data_range": 1.0,
    "boundary": "reflect",
    "filter": "scipy.ndimage.uniform_filter",
}


def windowed_ssim(a, b):
    """Mean windowed SSIM between two 2D arrays already in [0, 1]."""
    import numpy as np
    from scipy.ndimage import uniform_filter
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("shape mismatch %r vs %r" % (a.shape, b.shape))
    win = WINDOWED_SSIM_PARAMS["win_size"]
    dr = WINDOWED_SSIM_PARAMS["data_range"]
    c1 = (WINDOWED_SSIM_PARAMS["K1"] * dr) ** 2
    c2 = (WINDOWED_SSIM_PARAMS["K2"] * dr) ** 2
    kw = {"size": win, "mode": "reflect"}

    mu_a = uniform_filter(a, **kw)
    mu_b = uniform_filter(b, **kw)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a2 = uniform_filter(a * a, **kw) - mu_a2
    sigma_b2 = uniform_filter(b * b, **kw) - mu_b2
    sigma_ab = uniform_filter(a * b, **kw) - mu_ab

    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    ssim_map = num / den
    return float(ssim_map.mean())
