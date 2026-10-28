"""Hakem odakli ek analizler (bulutta veya yerelde calisir).
A) Mesafe tanisi: CV test istasyonu-en yakin egitim istasyonu mesafesi ve izgara hucresi-en yakin istasyon mesafesi;
   mesafe sinifina gore dis-kat gunluk R2 (LightGBM+IDW)
B) Capraz uydurulmus sapma duzeltmesi (katsayi 9 kattan, 10. kata uygulanir)
C) Maruziyet duyarliligi: duzeltmesiz, +/- istasyon ortalamasi RMSE, deprem bolgesi haric
D) 73 gunluk ornekleme ile tum 2025 ortalamasinin istasyonlarda karsilastirilmasi
E) Alternatif oncelik siralamasi (en yakin istasyondan >10 km nufus)
F) Harita modeli (111 ozellik) icin SHAP
Kullanim: python cbs_ek_analizler.py [kok_klasor] [--shap-yok]
"""
import sys, os, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

arg = [a for a in sys.argv[1:] if not a.startswith("--")]
KOK = Path(arg[0]) if arg else Path(__file__).resolve().parent.parent
ILCE_DOSYA = KOK / "veri" / "TUR_ADM2_OCHA_CODAB.geojson"  # OCHA COD-AB ilce sinirlari (HDX)
if not ILCE_DOSYA.exists():
    ILCE_DOSYA = KOK / "veri" / "geoBoundaries-TUR-ADM2.geojson"
SHAP = "--shap-yok" not in sys.argv
ISL = KOK / "islenmis"
CIKTI = ISL / "ek_analizler"
CIKTI.mkdir(exist_ok=True)
LAEA = "+proj=laea +lat_0=39 +lon_0=35 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
KIRL = ["PM10", "NO2", "SO2", "O3"]
ETIKET = {"PM10": "PM$_{10}$", "NO2": "NO$_2$", "SO2": "SO$_2$", "O3": "O$_3$"}
WHO = {"PM10": 15, "NO2": 10}
DEPREM = ["KAHRAMANMARAS", "HATAY", "ADIYAMAN", "OSMANIYE", "GAZIANTEP", "SANLIURFA", "MALATYA",
          "DIYARBAKIR", "ADANA", "KILIS", "ELAZIG"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600, "savefig.bbox": "tight"})
MUREKKEP = "#3d3d3a"
RENK = {"grid": "#3d3d3a", "istasyon": "#eb6834", "il": "#2a78d6"}


def xy(lon, lat):
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", LAEA, always_xy=True)
    x, y = t.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.c_[x, y] / 1000.0


def r2(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


rapor = []
panel = pd.read_parquet(ISL / "cbs_panel_istasyon_gunluk.parquet", columns=["istasyon_id", "tarih", "lon", "lat"] + KIRL)
ist_xy = panel.groupby("istasyon_id")[["lon", "lat"]].first()
H = pd.read_parquet(ISL / "izgara" / "H_harita_2025.parquet")
_N = KOK / "islenmis" / "izgara" / "N_nufus_2025.parquet"
if _N.exists():  # TUIK ilce nufusuna olceklenmis hucre nufusu (cbs_nufus_duzeltme.py)
    H["nufus_hucre"] = H["hucre_id"].map(pd.read_parquet(_N).set_index("hucre_id")["nufus_duz"]).values
w = H["nufus_hucre"].fillna(0).clip(lower=0).values
gxy = xy(H.lon, H.lat)

# ---------------- A) mesafe tanisi ----------------
satir_a, satir_b = [], []
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0), sharey=True, sharex=True)
axs = axs.ravel()
for i, k in enumerate(KIRL):
    ax = axs[i]
    ids = panel.loc[panel[k].notna()].groupby("istasyon_id").size()
    for sema in ["istasyon", "il"]:
        o = pd.read_parquet(ISL / "model" / "oof" / f"{k}_{sema}_lgbm_idw.parquet")
        kat = o.groupby("istasyon_id")["kat"].first()
        P = xy(ist_xy.loc[kat.index, "lon"], ist_xy.loc[kat.index, "lat"])
        d_test = np.full(len(kat), np.nan)
        for f in np.unique(kat.values):
            te = kat.values == f
            d_test[te] = cKDTree(P[~te]).query(P[te])[0]
        dd = pd.Series(d_test, index=kat.index)
        o["d"] = o["istasyon_id"].map(dd)
        # mesafe sinifina gore gunluk R2 (havuzlanmis)
        for lo, hi in [(0, 10), (10, 25), (25, 50), (50, 100), (100, 1e9)]:
            m = (o.d >= lo) & (o.d < hi)
            if m.sum() > 1000:
                satir_b.append({"kirletici": k, "sema": sema, "aralik_km": f"{lo}-{hi if hi < 1e8 else 'inf'}",
                                "istasyon": int(o.loc[m, "istasyon_id"].nunique()), "n": int(m.sum()),
                                "gunluk_R2": r2(o.loc[m, "gozlem"].values, o.loc[m, "tahmin"].values)})
        s = np.sort(d_test)
        ax.plot(s, np.arange(1, len(s) + 1) / len(s) * 100, color=RENK[sema], lw=1.1,
                label="Station-blocked: test to training" if sema == "istasyon" else "Province-blocked: test to training")
        satir_a.append({"kirletici": k, "dagilim": sema, "medyan_km": float(np.median(d_test)),
                        "p90_km": float(np.percentile(d_test, 90))})
    # izgara: hucre -> en yakin istasyon (nufus agirlikli ve alan)
    Sxy = xy(ist_xy.loc[ids.index, "lon"], ist_xy.loc[ids.index, "lat"])
    dg = cKDTree(Sxy).query(gxy)[0]
    o_ = np.argsort(dg)
    ax.plot(dg[o_], np.cumsum(w[o_]) / w.sum() * 100, color=RENK["grid"], lw=1.1, ls="-", label="Grid: population-weighted")
    ax.plot(np.sort(dg), np.arange(1, len(dg) + 1) / len(dg) * 100, color=RENK["grid"], lw=0.9, ls=":", label="Grid: area")
    satir_a.append({"kirletici": k, "dagilim": "izgara_nufus", "medyan_km": float(dg[o_][np.searchsorted(np.cumsum(w[o_]) / w.sum(), 0.5)]),
                    "p90_km": float(dg[o_][np.searchsorted(np.cumsum(w[o_]) / w.sum(), 0.9)])})
    satir_a.append({"kirletici": k, "dagilim": "izgara_alan", "medyan_km": float(np.median(dg)), "p90_km": float(np.percentile(dg, 90))})
    ax.set_xscale("log")
    ax.set_xlim(1, 300)
    if i >= 2:
        ax.set_xlabel("Distance to nearest station (km)")
    ax.set_title(f"({chr(97 + i)}) {ETIKET[k]}", loc="left")
    ax.grid(color="#e4e3dc", lw=0.5)
axs[0].set_ylabel("Cumulative share (%)")
axs[2].set_ylabel("Cumulative share (%)")
h, l = axs[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.savefig(ISL / "sonuc_sekilleri" / "sekil_S3_mesafe_tanisi.png")
plt.close(fig)
A = pd.DataFrame(satir_a)
B = pd.DataFrame(satir_b)
A.to_csv(CIKTI / "A_mesafe_dagilimi.csv", index=False)
B.to_csv(CIKTI / "A_mesafeye_gore_R2.csv", index=False)
rapor += ["A) Mesafe dagilimlari (medyan / p90 km):", A.round(1).to_string(index=False), "",
          "A) Mesafe sinifina gore gunluk R2:", B.round(3).to_string(index=False), ""]

# ---------------- B) capraz uydurulmus sapma duzeltmesi ----------------
satir = []
for k in KIRL:
    o = pd.read_parquet(ISL / "model" / "oof" / f"{k}_istasyon_lgbm_idw.parquet").dropna(subset=["gozlem", "tahmin"])
    c_all = o.gozlem.mean() / o.tahmin.mean()
    p_cf = o.tahmin.copy()
    for f in o.kat.unique():
        te = o.kat == f
        c = o.loc[~te, "gozlem"].mean() / o.loc[~te, "tahmin"].mean()
        p_cf[te] = o.loc[te, "tahmin"] * c
    st = lambda p: pd.DataFrame({"i": o.istasyon_id, "y": o.gozlem, "p": p}).groupby("i").mean()
    s0, s1, s2 = st(o.tahmin), st(o.tahmin * c_all), st(p_cf)
    satir.append({"kirletici": k, "c_tum": c_all, "R2_once": r2(o.gozlem.values, o.tahmin.values),
                  "R2_tum": r2(o.gozlem.values, (o.tahmin * c_all).values), "R2_capraz": r2(o.gozlem.values, p_cf.values),
                  "MB_once": (o.tahmin - o.gozlem).mean(), "MB_capraz": (p_cf - o.gozlem).mean(),
                  "istort_R2_once": r2(s0.y.values, s0.p.values), "istort_R2_tum": r2(s1.y.values, s1.p.values),
                  "istort_R2_capraz": r2(s2.y.values, s2.p.values)})
Bc = pd.DataFrame(satir)
Bc.to_csv(CIKTI / "B_capraz_sapma.csv", index=False)
rapor += ["B) Capraz uydurulmus sapma duzeltmesi (karsilastirma modeli LightGBM+IDW, istasyon gruplu):", Bc.round(3).to_string(index=False), ""]

# ---------------- C) maruziyet duyarliligi ----------------
V = pd.read_csv(ISL / "izgara" / "V_dogrulama.csv", encoding="utf-8-sig").set_index("kirletici")
M = pd.read_csv(ISL / "model" / "metrik_ozet.csv", encoding="utf-8-sig")
deprem = H.il_anahtar.isin(DEPREM).values
satir = []
for k in KIRL:
    v = H[f"{k}_yillik"].values
    c = V.loc[k, "duzeltme_katsayisi"]
    rm = M.query("kirletici==@k and sema=='istasyon' and model=='lgbm_idw'")["istasyon_ort_RMSE"].iloc[0]
    pw = lambda vv, ww: (vv * ww).sum() / ww.sum()
    r = {"kirletici": k, "pw_ana": pw(v, w), "pw_duzeltmesiz": pw(v / c, w), "pw_deprem_haric": pw(v[~deprem], w[~deprem]),
         "istasyon_ort_RMSE": rm}
    if k in WHO:
        e = WHO[k]
        f = lambda vv, ww: 100 * ww[vv > e].sum() / ww.sum()
        r.update({"WHO_ana": f(v, w), "WHO_duzeltmesiz": f(v / c, w), "WHO_eksi_RMSE": f(v - rm, w), "WHO_arti_RMSE": f(v + rm, w),
                  "WHO_deprem_haric": f(v[~deprem], w[~deprem])})
    satir.append(r)
C = pd.DataFrame(satir)
C.to_csv(CIKTI / "C_maruziyet_duyarlilik.csv", index=False)
rapor += ["C) Maruziyet duyarliligi:", C.round(2).to_string(index=False),
          f"   deprem bolgesindeki nufus payi: %{100 * w[deprem].sum() / w.sum():.1f}", ""]

# ---------------- D) 73 gun ile tum yil ----------------
p25 = panel[(panel.tarih >= "2025-01-01") & (panel.tarih < "2026-01-01")]
gun73 = pd.date_range("2025-01-01", periods=73, freq="5D")
satir = []
for k in KIRL:
    a = p25.groupby("istasyon_id")[k].agg(["mean", "count"])
    b = p25[p25.tarih.isin(gun73)].groupby("istasyon_id")[k].agg(["mean", "count"])
    j = a.join(b, rsuffix="_73").query("count >= 200 and count_73 >= 40")
    fark = 100 * (j["mean_73"] - j["mean"]) / j["mean"]
    satir.append({"kirletici": k, "istasyon": len(j), "r": np.corrcoef(j["mean"], j["mean_73"])[0, 1],
                  "ort_mutlak_fark_pct": fark.abs().mean(), "medyan_fark_pct": fark.median(), "p95_mutlak_fark_pct": fark.abs().quantile(0.95)})
Dd = pd.DataFrame(satir)
Dd.to_csv(CIKTI / "D_73gun_tum_yil.csv", index=False)
rapor += ["D) 73 gun ornekleme ile tum yil (istasyonlar, 2025):", Dd.round(3).to_string(index=False), ""]

# ---------------- E) alternatif oncelik ----------------
it = pd.read_csv(ISL / "maruziyet" / "tablo_istasyon_yeri_onceligi.csv", encoding="utf-8-sig")
ilce_cols = pd.read_csv(ISL / "maruziyet" / "tablo_ilce_maruziyet.csv", encoding="utf-8-sig")
# hucre->ilce eslemesi maruziyet betigindeki gibi yeniden kurulur
import geopandas as gpd
ilce = gpd.read_file(ILCE_DOSYA).to_crs(LAEA)[["shapeName", "shapeID", "geometry"]]
hucre = gpd.GeoDataFrame(geometry=gpd.points_from_xy(H.lon, H.lat), crs=4326).to_crs(LAEA)
es = gpd.sjoin(hucre, ilce, how="left", predicate="within")
es = es[~es.index.duplicated()]
H["ilce_id"] = es["shapeID"].values
bos = H.ilce_id.isna().values
if bos.any():
    j = gpd.sjoin_nearest(hucre[bos], ilce, how="left")
    H.loc[bos, "ilce_id"] = j[~j.index.duplicated()]["shapeID"].values
satir = []
for k in ["PM10", "NO2"]:
    ids = panel.loc[(panel[k].notna()) & (panel.tarih >= "2025-01-01") & (panel.tarih < "2026-01-01")].istasyon_id.unique()
    Sxy = xy(ist_xy.loc[ids, "lon"], ist_xy.loc[ids, "lat"])
    dg = cKDTree(Sxy).query(gxy)[0]
    m = (dg > 10) & (H[f"{k}_yillik"].values > WHO[k])
    t = pd.Series(w * m).groupby(H.ilce_id.values).sum().sort_values(ascending=False)
    alt = t.head(10)
    ana = it[it.kirletici == k].head(10)
    ad = ilce.set_index("shapeID")["shapeName"]
    il_ad = H.groupby("ilce_id").il_anahtar.agg(lambda s: s.mode().iat[0])
    ortak = len(set(alt.index) & set(ana.ilce_id))
    for sira, (i_, v_) in enumerate(alt.items(), 1):
        satir.append({"kirletici": k, "sira": sira, "ilce": ad.get(i_, i_), "il": il_ad.get(i_, ""), "nufus_10km_otesi_WHO_ustu": v_})
    rapor.append(f"E) {k}: >10 km ve WHO ustu nufusa gore ilk 10 ilce; ana siralamayla ortak: {ortak}/10")
E = pd.DataFrame(satir)
E.to_csv(CIKTI / "E_alternatif_oncelik.csv", index=False)
rapor += [E.to_string(index=False), ""]

# ---------------- F) harita modeli SHAP ----------------
if SHAP:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cbs_izgara as ci
    yol = ci.Yol(str(KOK), False)
    cm = ci.modelleme_modulu(str(KOK))
    p, oz, _ = ci.egitim_paneli(yol, cm)
    satir = []
    for k in KIRL:
        df = ci.hedef_verisi(p, k, cm, False)
        kom = cm.Komsuluk(df, k)
        I, en = kom.idw(np.ones(len(kom.ist), dtype=bool))
        a1, a2 = kom.satira(df, I, en)
        X = df[oz].assign(komsu_idw=a1, en_yakin_istasyon_km=a2)
        y = np.log1p(df[k].values)
        m = cm.model_kur("lgbm_idw", False, False)
        m.fit(X, y)
        rs = np.random.RandomState(42).choice(len(X), size=min(30000, len(X)), replace=False)
        con = m.predict(X.iloc[rs], pred_contrib=True)[:, :-1]
        onem = pd.Series(np.abs(con).mean(0), index=X.columns).sort_values(ascending=False)
        onem.rename("ortalama_abs_shap").to_csv(CIKTI / f"F_{k}_shap_harita_modeli.csv")
        satir.append(f"F) {k} SHAP ilk 10: " + "; ".join(f"{a} {b:.3f}" for a, b in onem.head(10).items()))
        # istasyon/izgara en yakin mesafe dagilimi (ozellik uyumu)
        satir.append(f"   egitimde en_yakin_istasyon_km min/medyan: {np.nanmin(a2):.2f}/{np.nanmedian(a2):.1f}")
    rapor += satir

(CIKTI / "ek_analiz_raporu.txt").write_text("\n".join(rapor), encoding="utf-8")
print("\n".join(rapor))
