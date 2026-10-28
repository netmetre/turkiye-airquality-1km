# -*- coding: utf-8 -*-
"""
CBS makalesi - Adim 2: kesifsel mekansal analiz
Girdi : islenmis/cbs_panel_istasyon_gunluk.parquet, veri/geoBoundaries-TUR-ADM1.geojson
Cikti : islenmis/kesifsel/ altinda tablolar (csv) ve sekiller (png, 600 dpi)

1) Istasyon ozetleri: donem ortalamasi, kis (Ara-Oca-Sub) ve yaz (Haz-Tem-Agu) ortalamasi
2) Kuresel Moran's I (k=8 en yakin komsu, satir standardize, 999 permutasyon)
3) Yerel Getis-Ord Gi* sicak/soguk noktalar (999 permutasyon, p<0.05)
4) Yerel Moran (LISA) kume tipleri
5) Deneysel yari-variogram (istasyon ortalamalari, km)
6) Istasyon ortalamasi ile statik degiskenler arasi Spearman korelasyonu
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local
from esda.getisord import G_Local

warnings.filterwarnings("ignore")

KOK = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
PANEL = os.path.join(KOK, "islenmis", "cbs_panel_istasyon_gunluk.parquet")
SINIR = os.path.join(KOK, "veri", "geoBoundaries-TUR-ADM1.geojson")
CIKTI = os.path.join(KOK, "islenmis", "kesifsel")
os.makedirs(CIKTI, exist_ok=True)

KIRLETICI = ["PM10", "NO2", "SO2", "O3"]
BIRIM = {"PM10": "µg m$^{-3}$", "NO2": "µg m$^{-3}$", "SO2": "µg m$^{-3}$", "O3": "µg m$^{-3}$"}
GOSTER = {"PM10": "PM$_{10}$", "NO2": "NO$_2$", "SO2": "SO$_2$", "O3": "O$_3$"}
MIN_GUN = 365            # donem ortalamasi icin en az gecerli gun
MIN_MEVSIM_GUN = 90      # mevsim ortalamasi icin en az gecerli gun
K = 8
PERM = 999
SEED = 42
LAEA = "+proj=laea +lat_0=39 +lon_0=35 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "savefig.dpi": 600, "savefig.bbox": "tight",
})

p = pd.read_parquet(PANEL)
p["mevsim"] = np.select([p["ay"].isin([12, 1, 2]), p["ay"].isin([6, 7, 8])], ["kis", "yaz"], "diger")
statik_sutun = [c for c in p.columns if c.startswith(("wc_", "yol_", "nufus_", "yapilasma_", "sanayi_",
                "maden_", "uzaklik_", "yukseklik", "egim", "goreli_", "fosil_", "komur_", "gsa_", "gwa_"))]
yillik_sutun = [c for c in p.columns if c.startswith(("ndvi_", "gece_isigi_", "tuik_"))]
ist = (p.groupby("istasyon_id")
         .agg(istasyon=("istasyon", "first"), il=("il", "first"), lon=("lon", "first"), lat=("lat", "first"),
              tip=("istasyon_tip_kodu", "first"),
              **{c: (c, "first") for c in statik_sutun},
              **{c: (c, "mean") for c in yillik_sutun})
         .reset_index())

# ---------------------------------------------------------------- 1) ozetler
for k in KIRLETICI:
    g = p.groupby("istasyon_id")[k]
    n = g.count()
    ist = ist.merge(pd.DataFrame({f"{k}_ort": g.mean().where(n >= MIN_GUN), f"{k}_gun": n}),
                    left_on="istasyon_id", right_index=True, how="left")
    for m in ["kis", "yaz"]:
        gm = p[p["mevsim"] == m].groupby("istasyon_id")[k]
        ist = ist.merge(gm.mean().where(gm.count() >= MIN_MEVSIM_GUN).rename(f"{k}_{m}"),
                        left_on="istasyon_id", right_index=True, how="left")
ist.to_csv(os.path.join(CIKTI, "istasyon_ozet.csv"), index=False, encoding="utf-8-sig")

gdf = gpd.GeoDataFrame(ist, geometry=gpd.points_from_xy(ist.lon, ist.lat), crs="EPSG:4326").to_crs(LAEA)
sinir = gpd.read_file(SINIR).to_crs(LAEA)


def agirlik(sub):
    w = KNN.from_dataframe(sub, k=K)
    w.transform = "r"
    return w


# ---------------------------------------------------------------- 2-4) Moran, Gi*, LISA
moran_satir, yerel = [], {}
for k in KIRLETICI:
    for donem, sut in [("donem", f"{k}_ort"), ("kis", f"{k}_kis"), ("yaz", f"{k}_yaz")]:
        sub = gdf[gdf[sut].notna()].copy()
        if len(sub) < 30:
            continue
        y = sub[sut].values
        w = agirlik(sub)
        mi = Moran(y, w, permutations=PERM)
        ylog = np.log1p(y)
        mil = Moran(ylog, w, permutations=PERM)
        moran_satir.append({"kirletici": k, "donem": donem, "n_istasyon": len(sub),
                            "moran_I": mi.I, "beklenen_I": mi.EI, "z": mi.z_sim, "p_perm": mi.p_sim,
                            "moran_I_log": mil.I, "p_perm_log": mil.p_sim})
        if donem == "donem":
            np.random.seed(SEED)
            gi = G_Local(y, KNN.from_dataframe(sub, k=K), transform="B", permutations=PERM, star=True, seed=SEED)
            lm = Moran_Local(ylog, w, permutations=PERM, seed=SEED)
            sub["gi_z"] = gi.Zs
            sub["gi_p"] = gi.p_sim
            sub["gi_sinif"] = np.where((gi.p_sim < 0.05) & (gi.Zs > 0), "sicak",
                                       np.where((gi.p_sim < 0.05) & (gi.Zs < 0), "soguk", "anlamsiz"))
            etiket = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
            sub["lisa"] = [etiket[q] if s < 0.05 else "anlamsiz" for q, s in zip(lm.q, lm.p_sim)]
            yerel[k] = sub
moran = pd.DataFrame(moran_satir)
moran.to_csv(os.path.join(CIKTI, "tablo_moran.csv"), index=False, encoding="utf-8-sig")

yerel_tablo = []
for k, sub in yerel.items():
    t = sub[["istasyon_id", "istasyon", "il", f"{k}_ort", "gi_z", "gi_p", "gi_sinif", "lisa"]].copy()
    t.columns = ["istasyon_id", "istasyon", "il", "ortalama", "gi_z", "gi_p", "gi_sinif", "lisa"]
    t.insert(0, "kirletici", k)
    yerel_tablo.append(t)
pd.concat(yerel_tablo).to_csv(os.path.join(CIKTI, "tablo_yerel_gi_lisa.csv"), index=False, encoding="utf-8-sig")

sicak_il = []
for k, sub in yerel.items():
    s = sub[sub["gi_sinif"] == "sicak"].groupby("il").size().sort_values(ascending=False)
    c = sub[sub["gi_sinif"] == "soguk"].groupby("il").size().sort_values(ascending=False)
    sicak_il.append({"kirletici": k, "sicak_istasyon": int(s.sum()), "soguk_istasyon": int(c.sum()),
                     "sicak_iller": ", ".join(f"{i} ({n})" for i, n in s.head(8).items()),
                     "soguk_iller": ", ".join(f"{i} ({n})" for i, n in c.head(8).items())})
pd.DataFrame(sicak_il).to_csv(os.path.join(CIKTI, "tablo_sicak_soguk_iller.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 5) variogram
def variogram(sub, sut, kutular):
    xy = np.c_[sub.geometry.x, sub.geometry.y] / 1000.0
    v = sub[sut].values
    i, j = np.triu_indices(len(v), 1)
    d = np.hypot(*(xy[i] - xy[j]).T)
    g = 0.5 * (v[i] - v[j]) ** 2
    idx = np.digitize(d, kutular)
    out = []
    for b in range(1, len(kutular)):
        m = idx == b
        if m.sum() >= 30:
            out.append({"mesafe_km": d[m].mean(), "yari_varyans": g[m].mean(), "cift": int(m.sum())})
    return pd.DataFrame(out)


kutular = np.r_[0, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 600, 800]
vg = []
for k in KIRLETICI:
    sub = gdf[gdf[f"{k}_ort"].notna()]
    t = variogram(sub, f"{k}_ort", kutular)
    t.insert(0, "kirletici", k)
    t["varyans"] = sub[f"{k}_ort"].var()
    vg.append(t)
vg = pd.concat(vg)
vg.to_csv(os.path.join(CIKTI, "tablo_variogram.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 6) korelasyon
aday = [c for c in statik_sutun + yillik_sutun if ist[c].nunique() > 5]
kor = []
for k in KIRLETICI:
    sub = ist[ist[f"{k}_ort"].notna()]
    for c in aday:
        d = sub[[f"{k}_ort", c]].dropna()
        if len(d) < 30:
            continue
        r, pv = spearmanr(d.iloc[:, 0], d.iloc[:, 1])
        kor.append({"kirletici": k, "degisken": c, "rho": r, "p": pv, "n": len(d)})
kor = pd.DataFrame(kor)
kor["abs_rho"] = kor["rho"].abs()
kor.sort_values(["kirletici", "abs_rho"], ascending=[True, False]).to_csv(
    os.path.join(CIKTI, "tablo_spearman_statik.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- sekiller
def harita_ekseni(ax):
    sinir.boundary.plot(ax=ax, color="0.55", linewidth=0.3, zorder=1)
    ax.set_axis_off()
    ax.set_aspect("equal")


# Sekil 1: istasyon donem ortalamalari
fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.6))
for ax, k, harf in zip(axs.ravel(), KIRLETICI, "abcd"):
    harita_ekseni(ax)
    sub = gdf[gdf[f"{k}_ort"].notna()].sort_values(f"{k}_ort")
    vmax = np.nanpercentile(sub[f"{k}_ort"], 98)
    sc = ax.scatter(sub.geometry.x, sub.geometry.y, c=sub[f"{k}_ort"], s=9, cmap="viridis",
                    vmin=sub[f"{k}_ort"].min(), vmax=vmax, edgecolors="k", linewidths=0.15, zorder=2)
    cb = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.01)
    cb.set_label(f"{GOSTER[k]} ({BIRIM[k]})")
    ax.set_title(f"({harf}) {GOSTER[k]}, n = {len(sub)}", loc="left")
fig.tight_layout()
fig.savefig(os.path.join(CIKTI, "sekil_1_istasyon_ortalamalari.png"))
plt.close(fig)

# Sekil 2: Gi* sicak/soguk noktalar
renk = {"sicak": "#c0392b", "soguk": "#2c6fbb", "anlamsiz": "0.8"}
fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.6))
for ax, k, harf in zip(axs.ravel(), KIRLETICI, "abcd"):
    harita_ekseni(ax)
    sub = yerel[k]
    for s_ad, s_et in [("anlamsiz", "Not significant"), ("soguk", "Cold spot"), ("sicak", "Hot spot")]:
        t = sub[sub["gi_sinif"] == s_ad]
        ax.scatter(t.geometry.x, t.geometry.y, s=9 if s_ad != "anlamsiz" else 6, c=renk[s_ad],
                   edgecolors="k" if s_ad != "anlamsiz" else "none", linewidths=0.15, zorder=3, label=s_et)
    ax.set_title(f"({harf}) {GOSTER[k]}", loc="left")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(CIKTI, "sekil_2_gi_star.png"))
plt.close(fig)

# Sekil 3: variogram
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0), sharex=True)
for ax, k, harf in zip(axs, KIRLETICI, "abcd"):
    t = vg[vg["kirletici"] == k]
    ax.plot(t["mesafe_km"], t["yari_varyans"], "o-", color="k", ms=3, lw=0.8)
    ax.axhline(t["varyans"].iloc[0], color="0.5", ls="--", lw=0.7)
    ax.set_xlabel("Distance (km)")
    ax.set_title(f"({harf}) {GOSTER[k]}", loc="left")
    ax.set_xscale("log")
    ax.set_xlim(2, 1000)
    ax.set_ylim(0, 1.12 * max(t["yari_varyans"].max(), t["varyans"].iloc[0]))
axs[0].set_ylabel("Semivariance (µg$^2$ m$^{-6}$)")
fig.tight_layout()
fig.savefig(os.path.join(CIKTI, "sekil_3_variogram.png"))
plt.close(fig)

# ---------------------------------------------------------------- ozet
r = ["KESIFSEL MEKANSAL ANALIZ OZETI", "=" * 60]
r.append(f"Istasyon sayisi (donem ortalamasi, >= {MIN_GUN} gun): " +
         ", ".join(f"{k} {int(ist[f'{k}_ort'].notna().sum())}" for k in KIRLETICI))
r.append("")
r.append(f"Kuresel Moran's I (k={K} komsu, {PERM} permutasyon):")
for _, s in moran.iterrows():
    r.append(f"  {s.kirletici:<5} {s.donem:<6} n={s.n_istasyon:<4} I={s.moran_I:+.3f} z={s.z:+.2f} "
             f"p={s.p_perm:.3f}   log: I={s.moran_I_log:+.3f} p={s.p_perm_log:.3f}")
r.append("")
r.append("Gi* sicak/soguk nokta (p<0.05):")
for s in sicak_il:
    r.append(f"  {s['kirletici']}: sicak {s['sicak_istasyon']} [{s['sicak_iller']}]")
    r.append(f"  {'':5} soguk {s['soguk_istasyon']} [{s['soguk_iller']}]")
r.append("")
r.append("Istasyon ortalamasiyla en guclu 6 statik/yillik degisken (Spearman):")
for k in KIRLETICI:
    t = kor[kor["kirletici"] == k].sort_values("abs_rho", ascending=False).head(6)
    r.append(f"  {k}: " + "; ".join(f"{a} {b:+.2f}" for a, b in zip(t["degisken"], t["rho"])))
with open(os.path.join(CIKTI, "kesifsel_ozet.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(r) + "\n")
print("\n".join(r))
