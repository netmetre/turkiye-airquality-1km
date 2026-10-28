"""CBS makalesi sonuç şekilleri (modelleme çıktılarından).

Girdi: islenmis/model (metrik_ozet, metrik_kat, shap, oof), islenmis/kesifsel/istasyon_ozet.csv,
veri/geoBoundaries-TUR-ADM1.geojson. Çıktı: islenmis/sonuc_sekilleri/.
Kullanım: python cbs_sonuc_sekilleri.py [kok_klasor]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

KOK = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent  # betikler kod\ altinda
MODEL = KOK / "islenmis" / "model"
CIKTI = KOK / "islenmis" / "sonuc_sekilleri"
CIKTI.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.dpi": 600, "savefig.bbox": "tight",
})

KIRL = ["PM10", "NO2", "SO2", "O3"]
KIRL_ETIKET = {"PM10": "PM$_{10}$", "NO2": "NO$_2$", "SO2": "SO$_2$", "O3": "O$_3$"}
MODELLER = ["idw", "ridge", "rf", "xgb", "lgbm", "lgbm_idw", "gnn"]
MODEL_ETIKET = {"idw": "IDW", "ridge": "Ridge LUR", "rf": "RF", "xgb": "XGBoost",
                "lgbm": "LightGBM", "lgbm_idw": "LightGBM+IDW", "gnn": "GNN"}
# Şema renkleri tüm şekillerde aynı
RENK = {"il": "#2a78d6", "istasyon": "#eb6834", "rastgele": "#1baf7a"}
SEMA_ETIKET = {"il": "Province-blocked", "istasyon": "Station-blocked", "rastgele": "Random"}
MUREKKEP = "#3d3d3a"
IZGARA = "#e4e3dc"


def panel_etiket(ax, i, kirl):
    ax.set_title(f"({chr(97 + i)}) {KIRL_ETIKET[kirl]}", loc="left", pad=4)


def izgara(ax, eksen="y"):
    ax.grid(axis=eksen, color=IZGARA, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


ozet = pd.read_csv(MODEL / "metrik_ozet.csv", encoding="utf-8-sig")
kat = pd.read_csv(MODEL / "metrik_kat.csv", encoding="utf-8-sig")

# ---------------- Şekil 4: model karşılaştırması (günlük R2) ----------------
fig, axs = plt.subplots(2, 2, figsize=(6.9, 4.6), sharey=True)
x = np.arange(len(MODELLER))
g = 0.38
for i, (ax, k) in enumerate(zip(axs.flat, KIRL)):
    d = ozet[ozet.kirletici == k].set_index(["sema", "model"])["gunluk_R2"]
    for j, s in enumerate(["il", "istasyon"]):
        v = [d.get((s, m), np.nan) for m in MODELLER]
        ax.bar(x + (j - 0.5) * g, v, g * 0.92, color=RENK[s], zorder=2,
               label=SEMA_ETIKET[s])
    ax.axhline(0, color=MUREKKEP, lw=0.6)
    ax.set_xticks(x, [MODEL_ETIKET[m] for m in MODELLER], rotation=35, ha="right")
    izgara(ax)
    panel_etiket(ax, i, k)
for ax in axs[:, 0]:
    ax.set_ylabel("Daily $R^2$")
axs[0, 0].legend(frameon=False, loc="upper left", ncol=2)
for ax in axs[0]:
    ax.tick_params(labelbottom=False)
fig.tight_layout(h_pad=1.0)
fig.savefig(CIKTI / "sekil_4_model_karsilastirma.png")
plt.close(fig)

# ---------------- Şekil 5: rastgele bölmenin şişirmesi ----------------
fig, ax = plt.subplots(figsize=(4.2, 2.6))
x = np.arange(len(KIRL))
g = 0.26
for j, s in enumerate(["rastgele", "istasyon", "il"]):
    v = [ozet.query("kirletici==@k and sema==@s and model=='lgbm'")["gunluk_R2"].iloc[0] for k in KIRL]
    b = ax.bar(x + (j - 1) * g, v, g * 0.92, color=RENK[s], label=SEMA_ETIKET[s], zorder=2)
    for xi, vi in zip(x + (j - 1) * g, v):
        ax.text(xi, vi + 0.012, f"{vi:.2f}", ha="center", va="bottom", fontsize=6.5, color=MUREKKEP)
ax.set_xticks(x, [KIRL_ETIKET[k] for k in KIRL])
ax.set_ylabel("Daily $R^2$ (LightGBM)")
ax.set_ylim(0, 0.88)
izgara(ax)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
fig.savefig(CIKTI / "sekil_5_rastgele_sisirme.png")
plt.close(fig)

# ---------------- Şekil 6: kat bazında R2 dağılımı ----------------
ALT, UST = -0.35, 0.62   # aşırı uç katlar alt sınırda üçgenle gösterilir
fig, axs = plt.subplots(2, 2, figsize=(6.9, 4.8), sharey=True)
for i, (ax, k) in enumerate(zip(axs.flat, KIRL)):
    for j, s in enumerate(["il", "istasyon"]):
        veri = [kat.query("kirletici==@k and sema==@s and model==@m")["R2"].values for m in MODELLER]
        pos = np.arange(len(MODELLER)) + (j - 0.5) * 0.38
        bp = ax.boxplot(veri, positions=pos, widths=0.32, patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", lw=1.0),
                        whiskerprops=dict(color=RENK[s], lw=0.7), capprops=dict(color=RENK[s], lw=0.7),
                        boxprops=dict(lw=0))
        for p in bp["boxes"]:
            p.set_facecolor(RENK[s])
        for q, (v, xp) in enumerate(zip(veri, pos)):
            alt_b = bp["whiskers"][2 * q].get_ydata().min()
            ust_b = bp["whiskers"][2 * q + 1].get_ydata().max()
            uc = v[(v < alt_b) | (v > ust_b)]
            ic = uc[uc >= ALT]
            ax.plot([xp] * len(ic), ic, "o", ms=2.5, mfc=RENK[s], mec="none")
            n_dis = int((uc < ALT).sum())
            if n_dis:
                ax.plot(xp, ALT + 0.02, "v", ms=6, mfc=RENK[s], mec=MUREKKEP, mew=0.4, clip_on=False, zorder=5)
    ax.set_ylim(ALT, UST)
    ax.axhline(0, color=MUREKKEP, lw=0.6)
    ax.set_xticks(np.arange(len(MODELLER)), [MODEL_ETIKET[m] for m in MODELLER], rotation=35, ha="right")
    izgara(ax)
    panel_etiket(ax, i, k)
for ax in axs[:, 0]:
    ax.set_ylabel("Fold $R^2$")
for ax in axs[0]:
    ax.tick_params(labelbottom=False)
axs[0, 0].legend(handles=[Patch(color=RENK[s], label=SEMA_ETIKET[s]) for s in ["il", "istasyon"]],
                 frameon=False, loc="lower left", ncol=2)
fig.tight_layout(h_pad=1.0)
fig.savefig(CIKTI / "sekil_6_kat_dagilimi.png")
plt.close(fig)

# ---------------- Şekil 7: SHAP önem sıralaması ----------------
OZ_ETIKET = {
    "komsu_idw": "Neighbor IDW", "doy_cos": "Day of year (cos)", "yilin_gunu": "Day of year",
    "yil": "Year", "era5_ruzgar_v_ms": "Wind v-component", "era5_ruzgar_hiz_ms": "Wind speed",
    "era5_sicaklik_c": "Air temperature", "era5_yagis_mm": "Precipitation",
    "era5_gunes_mj_m2": "Solar radiation", "era5_bagil_nem_pct": "Relative humidity",
    "yol_diger_km_1000": "Minor road length (1 km)", "yol_toplam_km_500": "Road length (500 m)",
    "uzaklik_ana_yol_m": "Distance to major road", "wc_yapi_500": "Built-up cover (500 m)",
    "gece_isigi_1000": "Night lights (1 km)", "nufus_piksel": "Population (pixel)",
    "s5p_no2_mol_m2": "S5P NO$_2$ column", "goreli_yukseklik_5km": "Relative elevation (5 km)",
    "uzaklik_komur_santral_m": "Distance to coal plant", "ndvi_1000": "NDVI (1 km)",
    "ndvi_3000": "NDVI (3 km)", "gwa_ruzgar_100m_ms": "Mean wind at 100 m",
    "sanayi_oran_3000": "Industrial land (3 km)", "tuik_kisi_basi_gsyh_usd": "GDP per capita",
    "tuik_ucak_inis_kalkis": "Aircraft movements",
}


def oz_etiket(o):
    import re
    if o in OZ_ETIKET:
        return OZ_ETIKET[o]
    m = re.match(r"(.*)_(500|1000|3000|5000)$", o)
    tampon = f" ({int(m.group(2)) / 1000:g} km)" if m else ""
    kok = m.group(1) if m else o
    WC = {"agac": "Tree cover", "cali": "Shrubland", "cayir": "Grassland", "tarim": "Cropland", "yapi": "Built-up cover",
          "ciplak": "Bare land", "kar": "Snow and ice", "su": "Water", "sulak": "Wetland"}
    G = {"gece_isigi": "Night lights", "ndvi": "NDVI", "nufus": "Population", "yapilasma_oran": "Built-up fraction",
         "yol_ana_km": "Major road length", "yol_orta_km": "Secondary road length", "yol_diger_km": "Minor road length",
         "yol_toplam_km": "Road length", "sanayi_oran": "Industrial land", "maden_ocagi_oran": "Quarry land",
         "en_yakin_istasyon_km": "Distance to nearest station", "era5_cig_noktasi_c": "Dew point",
         "era5_basinc_hpa": "Surface pressure", "era5_ruzgar_u_ms": "Wind u-component", "lon": "Longitude", "lat": "Latitude",
         "ay": "Month", "haftanin_gunu": "Day of week", "doy_sin": "Day of year (sin)", "yukseklik_m": "Elevation",
         "egim_derece": "Slope", "yukseklik_ort_1km": "Mean elevation (1 km)", "yukseklik_ort_5km": "Mean elevation (5 km)",
         "yukseklik_std_5km": "Elevation SD (5 km)", "uzaklik_sanayi_m": "Distance to industry",
         "uzaklik_fosil_santral_m": "Distance to fossil plant", "fosil_kapasite_mw_10km": "Fossil capacity (10 km)",
         "fosil_kapasite_mw_25km": "Fossil capacity (25 km)", "komur_kapasite_mw_25km": "Coal capacity (25 km)",
         "gsa_ghi_kwh_m2_gun": "Long-term irradiation", "gsa_sicaklik_uzun_donem_c": "Long-term temperature",
         "s5p_so2_mol_m2": "S5P SO$_2$ column", "modis_aod550": "MAIAC AOD"}
    if kok.startswith("wc_"):
        return WC.get(kok[3:], kok) + tampon
    if kok in G:
        return G[kok] + tampon
    return o.replace("_", " ")


N_UST = 10
fig, axs = plt.subplots(2, 2, figsize=(6.9, 5.0))
for i, (ax, k) in enumerate(zip(axs.flat, KIRL)):
    harita_shap = KOK / "islenmis" / "ek_analizler" / f"F_{k}_shap_harita_modeli.csv"
    kaynak = harita_shap if harita_shap.exists() else MODEL / "shap" / f"{k}_shap_onem.csv"
    d = pd.read_csv(kaynak, encoding="utf-8-sig")
    d.columns = ["ozellik", "ortalama_abs_shap"]
    d = d.head(N_UST)[::-1]
    ax.barh(np.arange(len(d)), d.ortalama_abs_shap, 0.7, color=RENK["istasyon"], zorder=2)
    ax.set_yticks(np.arange(len(d)), [oz_etiket(o) for o in d.ozellik])
    ax.set_xlabel("Mean |SHAP| (log(1 + C) units)")
    izgara(ax, "x")
    panel_etiket(ax, i, k)
fig.tight_layout(w_pad=1.5, h_pad=1.2)
fig.savefig(CIKTI / "sekil_7_shap_onem.png")
plt.close(fig)

# ---------------- Şekil 8 ve 9: istasyon düzeyi ----------------
MIN_GUN = 90


def istasyon_ozet(k, m):
    d = pd.read_parquet(MODEL / "oof" / f"{k}_istasyon_{m}.parquet")
    d = d.dropna(subset=["gozlem", "tahmin"])

    def f(g):
        if len(g) < MIN_GUN:
            return pd.Series({"n": len(g), "gozlem": np.nan, "tahmin": np.nan, "r2": np.nan})
        sse = ((g.gozlem - g.tahmin) ** 2).sum()
        sst = ((g.gozlem - g.gozlem.mean()) ** 2).sum()
        return pd.Series({"n": len(g), "gozlem": g.gozlem.mean(), "tahmin": g.tahmin.mean(),
                          "r2": 1 - sse / sst if sst > 0 else np.nan})
    return d.groupby("istasyon_id").apply(f, include_groups=False).dropna()


st = {(k, m): istasyon_ozet(k, m) for k in KIRL for m in ["lgbm_idw", "idw"]}

# Şekil 8: istasyon ortalaması gözlem-tahmin (istasyon dışarıda, LightGBM+IDW)
fig, axs = plt.subplots(2, 2, figsize=(6.0, 5.6))
for i, (ax, k) in enumerate(zip(axs.flat, KIRL)):
    d = st[(k, "lgbm_idw")]
    lim = [0, max(d.gozlem.max(), d.tahmin.max()) * 1.05]
    ax.plot(lim, lim, color=MUREKKEP, lw=0.6, ls="--", zorder=1)
    ax.scatter(d.gozlem, d.tahmin, s=9, color=RENK["istasyon"], edgecolors="white", linewidths=0.3, zorder=2)
    r2 = 1 - ((d.gozlem - d.tahmin) ** 2).sum() / ((d.gozlem - d.gozlem.mean()) ** 2).sum()
    ax.text(0.04, 0.95, f"$R^2$ = {r2:.2f}\nn = {len(d)}", transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("Observed mean (µg m$^{-3}$)")
    ax.set_ylabel("Predicted mean (µg m$^{-3}$)")
    izgara(ax, "both")
    panel_etiket(ax, i, k)
fig.tight_layout(w_pad=1.5, h_pad=1.2)
fig.savefig(CIKTI / "sekil_8_istasyon_ortalama.png")
plt.close(fig)

# Şekil 9: istasyon bazında R2 kazancı (LightGBM+IDW eksi IDW), harita
import geopandas as gpd
iz = pd.read_csv(KOK / "islenmis" / "kesifsel" / "istasyon_ozet.csv", encoding="utf-8-sig",
                 usecols=["istasyon_id", "lon", "lat"]).drop_duplicates("istasyon_id").set_index("istasyon_id")
sinir = gpd.read_file(KOK / "veri" / "geoBoundaries-TUR-ADM1.geojson")
CRS = "+proj=laea +lat_0=39 +lon_0=35 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
sinir = sinir.to_crs(CRS)
cmap = LinearSegmentedColormap.from_list("ayrisan", ["#eb6834", "#f2f1ec", "#2a78d6"])
norm = TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=0.4)
fig, axs = plt.subplots(2, 2, figsize=(6.9, 3.9))
for i, (ax, k) in enumerate(zip(axs.flat, KIRL)):
    a, b = st[(k, "lgbm_idw")], st[(k, "idw")]
    fark = (a.r2 - b.r2).dropna().clip(-0.4, 0.4).to_frame("fark").join(iz, how="inner")
    nok = gpd.GeoDataFrame(fark, geometry=gpd.points_from_xy(fark.lon, fark.lat), crs=4326).to_crs(CRS)
    sinir.plot(ax=ax, color="#fafaf8", edgecolor="#b5b4ad", lw=0.25)
    nok = nok.sort_values("fark", key=np.abs)
    sc = ax.scatter(nok.geometry.x, nok.geometry.y, c=nok.fark, cmap=cmap, norm=norm, s=8,
                    edgecolors=MUREKKEP, linewidths=0.2, zorder=3)
    kazan = (a.r2 - b.r2).dropna()
    ax.text(0.01, 0.02, f"gain > 0 at {100 * (kazan > 0).mean():.0f}% of {len(kazan)} stations",
            transform=ax.transAxes, fontsize=7, color=MUREKKEP)
    ax.set_axis_off()
    panel_etiket(ax, i, k)
fig.tight_layout(w_pad=0.5, h_pad=0.6)
cb = fig.colorbar(sc, ax=axs, orientation="horizontal", fraction=0.04, pad=0.03, aspect=40,
                  extend="both")
cb.set_label("Station daily $R^2$ gain over IDW (LightGBM+IDW minus IDW)")
cb.outline.set_linewidth(0.4)
fig.savefig(CIKTI / "sekil_9_istasyon_kazanc_harita.png")
plt.close(fig)

# Tablo: makale için özet (il ve istasyon şeması, günlük ve istasyon ortalaması R2)
t = ozet[ozet.sema != "rastgele"].pivot_table(index=["kirletici", "model"], columns="sema",
                                               values=["gunluk_R2", "gunluk_RMSE", "istasyon_ort_R2"])
t.columns = [f"{a}_{b}" for a, b in t.columns]
t = t.reindex(pd.MultiIndex.from_product([KIRL, MODELLER])).round(3)
t.to_csv(CIKTI / "tablo_model_ozet.csv", encoding="utf-8-sig")
print("bitti:", sorted(p.name for p in CIKTI.iterdir()))
