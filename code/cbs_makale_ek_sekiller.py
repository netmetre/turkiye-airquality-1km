"""Makale icin ek sekiller: is akisi (Fig. 2), mevsimsel haritalar (Fig. S2), grafik ozet.
Kullanim: python cbs_makale_ek_sekiller.py [kok_klasor]
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import rasterio
import geopandas as gpd

KOK = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
IZG = KOK / "islenmis" / "izgara"
CIKTI = KOK / "islenmis" / "sonuc_sekilleri"
LAEA = "+proj=laea +lat_0=39 +lon_0=35 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9, "savefig.dpi": 600, "savefig.bbox": "tight",
})
MUREKKEP = "#3d3d3a"
SIRALI = LinearSegmentedColormap.from_list("sirali", ["#f2f7fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
ETIKET = {"PM10": "PM$_{10}$", "NO2": "NO$_2$", "SO2": "SO$_2$", "O3": "O$_3$"}
sinir = gpd.read_file(KOK / "veri" / "geoBoundaries-TUR-ADM1.geojson").to_crs(LAEA)


def oku(ad):
    with rasterio.open(IZG / ad) as r:
        a = r.read(1, masked=True).astype("float64").filled(np.nan)
        b = r.bounds
    return a, (b.left, b.right, b.bottom, b.top)


# ---------------- Fig. 2: is akisi ----------------
fig, ax = plt.subplots(figsize=(7.0, 4.3))
ax.set_xlim(0, 100)
ax.set_ylim(0, 62)
ax.axis("off")
DOLGU = {"veri": "#eaf2fc", "islem": "#f2f1ec", "cikti": "#fdf0e9"}
KENAR = {"veri": "#2a78d6", "islem": "#5f5e5a", "cikti": "#eb6834"}


def kutu(x, y, w, h, baslik, satirlar, tur):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=DOLGU[tur], ec=KENAR[tur], lw=0.8))
    ax.text(x + w / 2, y + h - 1.6, baslik, ha="center", va="top", fontsize=8.5, fontweight="bold", color=MUREKKEP)
    ax.text(x + w / 2, y + h - 5.4, "\n".join(satirlar), ha="center", va="top", fontsize=7.2,
            color=MUREKKEP, linespacing=1.25)


def ok(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=MUREKKEP, lw=0.8, shrinkA=2, shrinkB=2))


# satir 1: veri
kutu(1, 44, 30, 17, "Monitoring data", ["354 stations, hourly, 2019-2026", "QC, daily means (≥18 h)",
                                        "PM$_{10}$, NO$_2$, SO$_2$, O$_3$"], "veri")
kutu(35, 44, 30, 17, "Dynamic predictors (daily)", ["Sentinel-5P NO$_2$/SO$_2$ columns",
                                                    "MAIAC AOD (1 km)", "ERA5-Land meteorology"], "veri")
kutu(69, 44, 30, 17, "Static predictors", ["SRTM, WorldCover, GHSL,", "VIIRS lights, MODIS NDVI",
                                          "OSM roads/industry, power plants",
                                          "Buffers 0.5, 1, 3, 5 km"], "veri")
# satir 2: islem
kutu(1, 22, 30, 17, "Exploratory analysis", ["Moran's I, Getis-Ord Gi*",
                                             "Empirical semivariograms", "Spearman screening"], "islem")
kutu(35, 22, 30, 17, "Model comparison", ["IDW, ridge LUR, RF, XGBoost,", "LightGBM, LightGBM+IDW, GNN",
                                          "Random / station / province CV"], "islem")
kutu(69, 22, 30, 17, "1 km mapping (2025)", ["Same-raster grid predictors",
                                             "5-member station bootstrap",
                                             "Out-of-fold bias correction"], "islem")
# satir 3: cikti
kutu(18, 1, 30, 16, "Interpretation", ["SHAP (TreeSHAP)", "Station-level skill gain over IDW"], "cikti")
kutu(52, 1, 30, 16, "Exposure and network gaps", ["Population-weighted means",
                                                   "WHO 2021 / EU 2030 exceedance",
                                                   "Priority unmonitored districts"], "cikti")
ok(16, 44, 16, 39)
ok(50, 44, 50, 39)
ok(84, 44, 84, 39)
ok(31, 30.5, 35, 30.5)
ok(65, 30.5, 69, 30.5)
ok(45, 22, 36, 17)
ok(84, 22, 72, 17)
fig.savefig(CIKTI / "sekil_2_is_akisi.png")
plt.close(fig)

# ---------------- Fig. S2: mevsimsel haritalar (DJF ve JJA) ----------------
fig, axs = plt.subplots(4, 2, figsize=(7.0, 8.6))
for i, k in enumerate(["PM10", "NO2", "SO2", "O3"]):
    a1, ext = oku(f"{k}_2025_DJF.tif")
    a2, _ = oku(f"{k}_2025_JJA.tif")
    lo, hi = np.nanpercentile(np.r_[a1[np.isfinite(a1)], a2[np.isfinite(a2)]], [1, 99])
    for j, (a, s) in enumerate([(a1, "DJF"), (a2, "JJA")]):
        ax = axs[i, j]
        im = ax.imshow(a, extent=ext, cmap=SIRALI, vmin=lo, vmax=hi, interpolation="nearest")
        sinir.boundary.plot(ax=ax, color="#5f5e5a", linewidth=0.2)
        ax.set_axis_off()
        ax.set_title(f"({chr(97 + 2 * i + j)}) {ETIKET[k]}, {s}", loc="left", pad=2)
    cb = fig.colorbar(im, ax=axs[i, :], orientation="vertical", fraction=0.025, pad=0.01, extend="both")
    cb.set_label("µg m$^{-3}$")
    cb.outline.set_linewidth(0.4)
fig.savefig(CIKTI / "sekil_S2_mevsimsel_harita.png")
plt.close(fig)

# ---------------- Grafik ozet (1328 x 531 oraninda) ----------------
fig = plt.figure(figsize=(13.28, 5.31))
a, ext = oku("NO2_2025_yillik.tif")
axm = fig.add_axes([0.26, 0.02, 0.50, 0.90])
lo, hi = np.nanpercentile(a, [1, 99])
axm.imshow(a, extent=ext, cmap=SIRALI, vmin=lo, vmax=hi, interpolation="nearest")
sinir.boundary.plot(ax=axm, color="#5f5e5a", linewidth=0.3)
axm.set_axis_off()
axm.set_title("1 km annual NO$_2$, 2025", fontsize=15, color=MUREKKEP)
axl = fig.add_axes([0.01, 0.06, 0.27, 0.88])
axl.axis("off")
axl.text(0.0, 0.98, "354 monitors + satellite,\nreanalysis and GIS layers", fontsize=15, va="top", color=MUREKKEP)
axl.text(0.0, 0.62, "Spatially blocked validation", fontsize=15, va="top", color=MUREKKEP, fontweight="bold")
axl.text(0.0, 0.50, "Random CV R$^2$ 0.61-0.76\nStation-blocked R$^2$ 0.15-0.45", fontsize=14, va="top",
         color=MUREKKEP)
axl.text(0.0, 0.20, "LightGBM + neighbor IDW\nbeats IDW and kriging", fontsize=14, va="top", color=MUREKKEP)
axr = fig.add_axes([0.76, 0.06, 0.23, 0.88])
axr.axis("off")
axr.text(0.0, 0.98, "Population exposure", fontsize=15, va="top", color=MUREKKEP, fontweight="bold")
axr.text(0.0, 0.84, "92% above WHO NO$_2$ guideline\n99.9% above WHO PM$_{10}$ guideline",
         fontsize=14, va="top", color=MUREKKEP)
axr.text(0.0, 0.50, "766 of 973 districts\nwithout an NO$_2$ monitor", fontsize=14, va="top", color=MUREKKEP)
axr.text(0.0, 0.22, "Priority districts ranked\nfor new stations", fontsize=14, va="top", color=MUREKKEP)
plt.rcParams["savefig.bbox"] = "standard"
fig.savefig(CIKTI / "grafik_ozet.png", dpi=200)
plt.close(fig)
print("bitti")
