"""CBS makalesi harita sekilleri (islenmis/izgara GeoTIFF'lerinden).
Cikti: islenmis/sonuc_sekilleri/sekil_10_yillik_harita.png, sekil_11_belirsizlik_harita.png
Kullanim: python cbs_harita_sekilleri.py [kok_klasor]
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rasterio
import geopandas as gpd

KOK = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent  # betikler kod\ altinda
IZG = KOK / "islenmis" / "izgara"
CIKTI = KOK / "islenmis" / "sonuc_sekilleri"
CIKTI.mkdir(parents=True, exist_ok=True)
LAEA = "+proj=laea +lat_0=39 +lon_0=35 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "savefig.dpi": 600, "savefig.bbox": "tight",
})
KIRL = ["PM10", "NO2", "SO2", "O3"]
ETIKET = {"PM10": "PM$_{10}$", "NO2": "NO$_2$", "SO2": "SO$_2$", "O3": "O$_3$"}
# tek tonlu sirali renk (acik -> koyu)
SIRALI = LinearSegmentedColormap.from_list("sirali", ["#f2f7fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
TURUNCU = LinearSegmentedColormap.from_list("belirsiz", ["#fdf3ee", "#f6b79a", "#eb6834", "#b8461d", "#7a2c10"])

sinir = gpd.read_file(KOK / "veri" / "geoBoundaries-TUR-ADM1.geojson").to_crs(LAEA)


def oku(ad):
    with rasterio.open(IZG / ad) as r:
        a = r.read(1, masked=True).astype("float64").filled(np.nan)
        b = r.bounds
    return a, (b.left, b.right, b.bottom, b.top)


def panel(ax, a, ext, cmap, vmin, vmax, etiket, birim, i, log=False):
    from matplotlib.colors import LogNorm, Normalize
    norm = LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(a, extent=ext, cmap=cmap, norm=norm, interpolation="nearest")
    sinir.boundary.plot(ax=ax, color="#5f5e5a", linewidth=0.2)
    ax.set_axis_off()
    ax.set_title(f"({chr(97 + i)}) {etiket}", loc="left", pad=2)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, aspect=35, extend="both")
    cb.set_label(birim)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4)
    if log:
        from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
        aday = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200]
        t = [v for v in aday if vmin <= v <= vmax]
        cb.ax.xaxis.set_major_locator(FixedLocator(t))
        cb.ax.xaxis.set_minor_locator(NullLocator())
        cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


# Sekil 10: 2025 yillik ortalama
fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.6))
for i, (ax, k) in enumerate(zip(axs.flat, KIRL)):
    a, ext = oku(f"{k}_2025_yillik.tif")
    lo, hi = np.nanpercentile(a, [0.5, 99.95])
    panel(ax, a, ext, SIRALI, lo, hi, ETIKET[k], "Annual mean (µg m$^{-3}$, log scale)", i, log=True)
fig.tight_layout(w_pad=0.8, h_pad=0.6)
fig.savefig(CIKTI / "sekil_10_yillik_harita.png")
plt.close(fig)

# Sekil 11: goreli belirsizlik (topluluk SD / ortalama)
fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.6))
for i, (ax, k) in enumerate(zip(axs.flat, KIRL)):
    m, ext = oku(f"{k}_2025_yillik.tif")
    s, _ = oku(f"{k}_2025_sd.tif")
    from scipy.ndimage import uniform_filter
    gec = np.isfinite(s)
    s5 = uniform_filter(np.where(gec, s, 0), 5) / np.maximum(uniform_filter(gec.astype(float), 5), 1e-9)
    m5 = uniform_filter(np.where(gec, m, 0), 5) / np.maximum(uniform_filter(gec.astype(float), 5), 1e-9)
    cv = np.where(gec, 100 * s5 / m5, np.nan)
    panel(ax, cv, ext, TURUNCU, 0, np.nanpercentile(cv, 99), ETIKET[k], "Ensemble SD / mean (%)", i)
fig.tight_layout(w_pad=0.8, h_pad=0.6)
fig.savefig(CIKTI / "sekil_11_belirsizlik_harita.png")
plt.close(fig)
print("bitti")
