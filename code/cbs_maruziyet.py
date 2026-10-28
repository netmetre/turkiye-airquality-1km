"""CBS makalesi - Adim 5: nufus maruziyeti, izlenmeyen ilceler ve istasyon yeri onceligi.
Girdi : islenmis/izgara/H_harita_2025.parquet, veri/TUR_ADM2_OCHA_CODAB.geojson (yoksa geoBoundaries ADM2), panel (istasyon konumlari)
Cikti : islenmis/maruziyet/ (tablolar) ve islenmis/sonuc_sekilleri/sekil_12, sekil_13
Kullanim: python cbs_maruziyet.py [kok_klasor]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree

KOK = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent  # betikler kod\ altinda
ILCE_DOSYA = KOK / "veri" / "TUR_ADM2_OCHA_CODAB.geojson"  # OCHA COD-AB ilce sinirlari (HDX)
if not ILCE_DOSYA.exists():
    ILCE_DOSYA = KOK / "veri" / "geoBoundaries-TUR-ADM2.geojson"
CIKTI = KOK / "islenmis" / "maruziyet"
SEKIL = KOK / "islenmis" / "sonuc_sekilleri"
CIKTI.mkdir(parents=True, exist_ok=True)
LAEA = "+proj=laea +lat_0=39 +lon_0=35 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
KIRL = ["PM10", "NO2", "SO2", "O3"]
ETIKET = {"PM10": "PM$_{10}$", "NO2": "NO$_2$", "SO2": "SO$_2$", "O3": "O$_3$"}
# Yillik ortalama esikleri (µg m-3): DSO 2021 hava kalitesi rehberi (AQG) ve ara hedefleri, AB 2030 sinir degerleri
ESIK = {
    "PM10": {"WHO AQG": 15, "WHO IT-4": 20, "WHO IT-3": 30, "WHO IT-2": 50, "WHO IT-1": 70, "EU 2030": 20},
    "NO2": {"WHO AQG": 10, "WHO IT-3": 20, "WHO IT-2": 30, "WHO IT-1": 40, "EU 2030": 20},
}
UZAKLIK_KM = [10, 25, 50]
YIL = 2025

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "savefig.dpi": 600, "savefig.bbox": "tight",
})
MUREKKEP = "#3d3d3a"


def agirlikli(v, w):
    m = np.isfinite(v) & (w > 0)
    return float((v[m] * w[m]).sum() / w[m].sum()) if m.any() else np.nan


# ------------------------------------------------------------------ veri
H = pd.read_parquet(KOK / "islenmis" / "izgara" / f"H_harita_{YIL}.parquet")
_N = KOK / "islenmis" / "izgara" / f"N_nufus_{YIL}.parquet"
if _N.exists():  # TUIK ilce nufusuna olceklenmis hucre nufusu (cbs_nufus_duzeltme.py)
    H["nufus_hucre"] = H["hucre_id"].map(pd.read_parquet(_N).set_index("hucre_id")["nufus_duz"]).values
H["nufus_hucre"] = H["nufus_hucre"].fillna(0).clip(lower=0)
w = H["nufus_hucre"].values
toplam_nufus = w.sum()
hucre = gpd.GeoDataFrame(H[["hucre_id"]], geometry=gpd.points_from_xy(H.lon, H.lat), crs=4326).to_crs(LAEA)
hx, hy = hucre.geometry.x.values, hucre.geometry.y.values

ilce = gpd.read_file(ILCE_DOSYA).to_crs(LAEA)
ilce = ilce[["shapeName", "shapeID", "geometry"]].rename(columns={"shapeName": "ilce", "shapeID": "ilce_id"})
es = gpd.sjoin(hucre, ilce, how="left", predicate="within")
es = es[~es.index.duplicated()]
H["ilce_id"] = es["ilce_id"].values
# kiyida poligon disinda kalan hucreler: en yakin ilce
bos = H["ilce_id"].isna().values
if bos.any():
    j = gpd.sjoin_nearest(hucre[bos], ilce, how="left")
    j = j[~j.index.duplicated()]
    H.loc[bos, "ilce_id"] = j["ilce_id"].values
ilce_il = H.groupby("ilce_id")["il_anahtar"].agg(lambda s: s.mode().iat[0])
ilce = ilce.merge(ilce_il.rename("il_anahtar"), left_on="ilce_id", right_index=True, how="left")

# 2025'te olcum yapan istasyonlar (kirletici bazinda)
p = pd.read_parquet(KOK / "islenmis" / "cbs_panel_istasyon_gunluk.parquet",
                    columns=["istasyon_id", "tarih", "lon", "lat"] + KIRL)
p = p[(p["tarih"] >= f"{YIL}-01-01") & (p["tarih"] < f"{YIL + 1}-01-01")]
ist = {}
for k in KIRL:
    s = p[p[k].notna()].groupby("istasyon_id")[["lon", "lat"]].first()
    g = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s.lon, s.lat), crs=4326).to_crs(LAEA)
    ist[k] = g
tum = pd.concat([ist[k][["geometry"]] for k in KIRL])
tum = tum[~tum.index.duplicated()]
ist["herhangi"] = gpd.GeoDataFrame(tum, geometry="geometry", crs=LAEA)

# ------------------------------------------------------------------ ulusal tablo
sat = []
for k in KIRL:
    v = H[f"{k}_yillik"].values
    sd = H[f"{k}_sd"].values
    r = {"kirletici": k, "alan_ortalamasi": np.nanmean(v), "nufus_agirlikli": agirlikli(v, w),
         "nufus_agirlikli_sd": agirlikli(sd, w),
         "nufus_p5": np.nan, "nufus_p95": np.nan}
    o = np.argsort(v)
    cw = np.cumsum(w[o]) / w.sum()
    r["nufus_p5"] = float(v[o][np.searchsorted(cw, 0.05)])
    r["nufus_medyan"] = float(v[o][np.searchsorted(cw, 0.5)])
    r["nufus_p95"] = float(v[o][np.searchsorted(cw, 0.95)])
    for ad, e in ESIK.get(k, {}).items():
        ust = v > e
        r[f"nufus_yuzde_ust_{ad}"] = 100 * w[ust].sum() / toplam_nufus
        # belirsizlik: topluluk ortalamasi +/- 1 SD ile esik ustu nufus araligi
        r[f"nufus_yuzde_ust_{ad}_alt"] = 100 * w[(v - sd) > e].sum() / toplam_nufus
        r[f"nufus_yuzde_ust_{ad}_ust"] = 100 * w[(v + sd) > e].sum() / toplam_nufus
    sat.append(r)
ulusal = pd.DataFrame(sat)
ulusal.to_csv(CIKTI / "tablo_ulusal_maruziyet.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------ il ve ilce tablolari
def grup_tablosu(anahtar):
    g = H.groupby(anahtar)
    t = pd.DataFrame({"nufus": g["nufus_hucre"].sum(), "hucre": g.size()})
    for k in KIRL:
        t[f"{k}_nufus_agirlikli"] = g.apply(lambda d: agirlikli(d[f"{k}_yillik"].values, d["nufus_hucre"].values),
                                            include_groups=False)
        for ad in ("WHO AQG",):
            if k in ESIK:
                e = ESIK[k][ad]
                t[f"{k}_nufus_ust_{ad}"] = g.apply(lambda d: d.loc[d[f"{k}_yillik"] > e, "nufus_hucre"].sum(),
                                                  include_groups=False)
    return t


il = grup_tablosu("il_anahtar")
il.to_csv(CIKTI / "tablo_il_maruziyet.csv", encoding="utf-8-sig")

it = grup_tablosu("ilce_id")
it = ilce.drop(columns="geometry").set_index("ilce_id").join(it)
for k in KIRL + ["herhangi"]:
    j = gpd.sjoin(ist[k][["geometry"]], ilce[["ilce_id", "geometry"]], how="left", predicate="within")
    it[f"istasyon_{k}"] = j.groupby("ilce_id").size().reindex(it.index).fillna(0).astype(int)

# hucrelerin en yakin istasyona uzakligi
for k in KIRL + ["herhangi"]:
    xy = np.c_[ist[k].geometry.x, ist[k].geometry.y]
    d, _ = cKDTree(xy).query(np.c_[hx, hy])
    H[f"uzaklik_{k}_km"] = d / 1000.0
it["ilce_merkez_uzaklik_herhangi_km"] = H.groupby("ilce_id").apply(
    lambda d: agirlikli(d["uzaklik_herhangi_km"].values, d["nufus_hucre"].values), include_groups=False)
it.to_csv(CIKTI / "tablo_ilce_maruziyet.csv", encoding="utf-8-sig")

# ------------------------------------------------------------------ izlenmeyen nufus
iz = []
for k in KIRL + ["herhangi"]:
    izsiz = it[f"istasyon_{k}"] == 0
    r = {"kirletici": k, "istasyon_sayisi": len(ist[k]),
         "istasyonsuz_ilce": int(izsiz.sum()), "istasyonlu_ilce": int((~izsiz).sum()),
         "istasyonsuz_ilce_nufus_yuzde": 100 * it.loc[izsiz, "nufus"].sum() / toplam_nufus}
    for km in UZAKLIK_KM:
        r[f"nufus_yuzde_{km}km_otesi"] = 100 * w[H[f"uzaklik_{k}_km"].values > km].sum() / toplam_nufus
    if k in KIRL:
        v = H[f"{k}_yillik"].values
        ic = H["ilce_id"].isin(it.index[izsiz]).values
        r["istasyonsuz_ilce_nufus_agirlikli"] = agirlikli(v[ic], w[ic])
        r["istasyonlu_ilce_nufus_agirlikli"] = agirlikli(v[~ic], w[~ic])
        if k in ESIK:
            e = ESIK[k]["WHO AQG"]
            r["istasyonsuz_ilce_WHO_ustu_nufus_milyon"] = w[ic & (v > e)].sum() / 1e6
    iz.append(r)
izlenme = pd.DataFrame(iz)
izlenme.to_csv(CIKTI / "tablo_izlenmeyen_nufus.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------ istasyon yeri onceligi
# Oncelik: istasyonu olmayan ilcelerde DSO AQG ustunde yasayan nufus x (nufus agirlikli konsantrasyon / AQG)
onc = []
for k in ["PM10", "NO2"]:
    e = ESIK[k]["WHO AQG"]
    t = it[it[f"istasyon_{k}"] == 0].copy()
    t["kirletici"] = k
    t["oncelik_puani"] = t[f"{k}_nufus_ust_WHO AQG"] * (t[f"{k}_nufus_agirlikli"] / e)
    t = t.sort_values("oncelik_puani", ascending=False)
    t["sira"] = np.arange(1, len(t) + 1)
    onc.append(t[["kirletici", "sira", "ilce", "il_anahtar", "nufus", f"{k}_nufus_agirlikli",
                  f"{k}_nufus_ust_WHO AQG", "ilce_merkez_uzaklik_herhangi_km", "oncelik_puani"]]
               .rename(columns={f"{k}_nufus_agirlikli": "nufus_agirlikli_konsantrasyon",
                                f"{k}_nufus_ust_WHO AQG": "WHO_AQG_ustu_nufus"}))
oncelik = pd.concat(onc)
oncelik.to_csv(CIKTI / "tablo_istasyon_yeri_onceligi.csv", encoding="utf-8-sig")

# ------------------------------------------------------------------ Sekil 12: kumulatif nufus dagilimi
RENK_ESIK = {"WHO AQG": "#2a78d6", "EU 2030": "#eb6834"}
fig, axs = plt.subplots(2, 2, figsize=(6.9, 4.6))
for i, (ax, k) in enumerate(zip(axs.flat, KIRL)):
    v = H[f"{k}_yillik"].values
    sd = H[f"{k}_sd"].values
    m = np.isfinite(v)
    for kayma, stil in [(0, "-"), (-1, ":"), (1, ":")]:
        vv = (v + kayma * sd)[m]
        o = np.argsort(vv)
        cw = np.cumsum(w[m][o]) / w[m].sum() * 100
        ax.plot(vv[o], cw, color=MUREKKEP, lw=1.2 if kayma == 0 else 0.7, ls=stil, zorder=3)
    for ad, e in ESIK.get(k, {}).items():
        if ad in RENK_ESIK:
            ax.axvline(e, color=RENK_ESIK[ad], lw=1.0, zorder=2, label=ad)
    oo = np.argsort(v); cc = np.cumsum(w[oo]) / w.sum()
    ax.set_xlim(0, v[oo][np.searchsorted(cc, 0.995)] * 1.1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Annual mean (µg m$^{-3}$)")
    ax.grid(color="#e4e3dc", lw=0.5, zorder=0)
    ax.set_title(f"({chr(97 + i)}) {ETIKET[k]}", loc="left")
    if k in ESIK:
        from matplotlib.lines import Line2D
        h, l = ax.get_legend_handles_labels()
        h += [Line2D([], [], color=MUREKKEP, lw=1.2), Line2D([], [], color=MUREKKEP, lw=0.7, ls=":")]
        l += ["Ensemble mean", "Mean ± 1 SD"]
        ax.legend(h, l, frameon=False, loc="lower right")
for ax in axs[:, 0]:
    ax.set_ylabel("Cumulative population (%)")
fig.tight_layout(h_pad=1.2)
fig.savefig(SEKIL / "sekil_12_nufus_maruziyet.png")
plt.close(fig)

# ------------------------------------------------------------------ Sekil 13: istasyonsuz ilcelerde oncelik
SIRALI = LinearSegmentedColormap.from_list("s", ["#fdf3ee", "#f6b79a", "#eb6834", "#b8461d", "#7a2c10"])
il_sinir = gpd.read_file(KOK / "veri" / "geoBoundaries-TUR-ADM1.geojson").to_crs(LAEA)
fig, axs = plt.subplots(2, 1, figsize=(6.0, 5.4))
for i, (ax, k) in enumerate(zip(axs, ["PM10", "NO2"])):
    t = ilce.merge(it[[f"istasyon_{k}", f"{k}_nufus_ust_WHO AQG"]], left_on="ilce_id", right_index=True)
    var = t[f"istasyon_{k}"] > 0
    t[~var].plot(ax=ax, column=t.loc[~var, f"{k}_nufus_ust_WHO AQG"] / 1000, cmap=SIRALI, linewidth=0,
                 legend=True, vmin=0, vmax=np.nanpercentile(t.loc[~var, f"{k}_nufus_ust_WHO AQG"] / 1000, 98),
                 legend_kwds={"label": "Population above WHO AQG\nin unmonitored districts (thousand)",
                              "shrink": 0.7, "pad": 0.01, "extend": "max"})
    t[var].plot(ax=ax, color="#d3d1c7", linewidth=0)
    il_sinir.boundary.plot(ax=ax, color="#5f5e5a", linewidth=0.25)
    s = ist[k]
    ax.scatter(s.geometry.x, s.geometry.y, s=2.5, color=MUREKKEP, zorder=4, linewidths=0)
    ax.set_axis_off()
    ax.set_title(f"({chr(97 + i)}) {ETIKET[k]}", loc="left")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend([Patch(color="#d3d1c7"), Line2D([], [], marker="o", ls="", ms=2.5, color=MUREKKEP)],
              ["District with a station", f"Station measuring {ETIKET[k]} in {YIL}"],
              frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.02), ncol=2, fontsize=7)
fig.tight_layout(h_pad=0.8)
fig.savefig(SEKIL / "sekil_13_istasyonsuz_ilce_oncelik.png")
plt.close(fig)

# ------------------------------------------------------------------ ozet
r = ["MARUZIYET OZETI (2025, 1 km)", "=" * 60, f"Toplam nufus (GHS-POP 2020): {toplam_nufus / 1e6:.1f} milyon", ""]
for _, x in ulusal.iterrows():
    r.append(f"{x.kirletici}: alan ort {x.alan_ortalamasi:.1f}, nufus agirlikli {x.nufus_agirlikli:.1f} "
             f"(medyan {x.nufus_medyan:.1f}, %5-%95 {x.nufus_p5:.1f}-{x.nufus_p95:.1f})")
    for ad in ESIK.get(x.kirletici, {}):
        r.append(f"    {ad} ustu nufus: %{x[f'nufus_yuzde_ust_{ad}']:.1f} "
                 f"(+/-1 SD: %{x[f'nufus_yuzde_ust_{ad}_alt']:.1f}-%{x[f'nufus_yuzde_ust_{ad}_ust']:.1f})")
r.append("")
for _, x in izlenme.iterrows():
    s = (f"{x.kirletici}: {x.istasyon_sayisi} istasyon, istasyonsuz ilce {x.istasyonsuz_ilce}/"
         f"{x.istasyonsuz_ilce + x.istasyonlu_ilce} (nufusun %{x.istasyonsuz_ilce_nufus_yuzde:.1f}), "
         + ", ".join(f">{km} km: %{x[f'nufus_yuzde_{km}km_otesi']:.1f}" for km in UZAKLIK_KM))
    if x.kirletici in KIRL:
        s += (f"; nufus agirlikli istasyonsuz {x.istasyonsuz_ilce_nufus_agirlikli:.1f} / "
              f"istasyonlu {x.istasyonlu_ilce_nufus_agirlikli:.1f}")
    r.append(s)
r.append("")
for k in ["PM10", "NO2"]:
    t = oncelik[oncelik.kirletici == k].head(10)
    r.append(f"{k} oncelikli istasyonsuz ilceler: " + "; ".join(f"{a} ({b}) {c / 1000:.0f} bin"
                                                               for a, b, c in t[["ilce", "il_anahtar", "nufus"]].values))
(CIKTI / "maruziyet_ozet.txt").write_text("\n".join(r) + "\n", encoding="utf-8")
print("\n".join(r))
