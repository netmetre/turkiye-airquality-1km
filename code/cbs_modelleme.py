# -*- coding: utf-8 -*-
"""
CBS makalesi - Adim 3: modelleme ve capraz dogrulama
=====================================================
Girdi : islenmis/cbs_panel_istasyon_gunluk.parquet
Cikti : islenmis/model/
          oof/{kirletici}_{sema}_{model}.parquet   katman disi (out-of-fold) tahminler
          metrik_kat.csv        kat bazinda metrikler
          metrik_ozet.csv       havuzlanmis gunluk ve istasyon-ortalamasi metrikleri
          kriging_ortalama.csv  ortalamalar uzerinde kriging sonucu
          shap/                 SHAP ozetleri
          model_log.txt, model_durum.json

Hedef  : istasyon gunluk ortalamasi (PM10, NO2, SO2, O3), log1p donusumu ile egitilir.
Semalar: rastgele (satir), istasyon (istasyon gruplari), il (il gruplari); 10 kat.
Modeller:
  idw        gunluk ters mesafe agirlikli enterpolasyon (k=10 en yakin egitim istasyonu)
  ridge      LUR: ridge regresyon
  rf         Random Forest
  xgb        XGBoost
  lgbm       LightGBM
  lgbm_idw   LightGBM + komsu istasyon IDW ozelligi (sizintisiz, kat icinde hesaplanir)
  gnn        indüktif graf sinir agi kriging (maskeli dugum egitimi)
Her (kirletici, sema, model) sonucu ayri dosyaya yazilir; dosya varsa atlanir.

Kullanim:
  python cbs_modelleme.py                       # hepsi
  python cbs_modelleme.py --kirletici NO2 --model lgbm,gnn
  python cbs_modelleme.py --hizli               # kucuk test
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

VARSAYILAN_KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
KIRLETICILER = ["PM10", "NO2", "SO2", "O3"]
SEMALAR = ["rastgele", "istasyon", "il"]
MODELLER = ["idw", "ridge", "rf", "xgb", "lgbm", "lgbm_idw", "gnn"]
RASTGELE_MODELLER = {"lgbm", "ridge"}      # rastgele bolmede yalniz bunlar (sisirmeyi gostermek icin)
KAT = 10
SEED = 42
MIN_GUN = 90                               # modele alinacak istasyonun en az gecerli gunu
IDW_K = 10
LAEA = "+proj=laea +lat_0=39 +lon_0=35 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

DISLA_ON_EK = ("PM10", "PM25", "SO2", "CO", "NO2", "NOX", "NO", "O3")
DISLA = {"istasyon_id", "tarih", "istasyon", "il", "il_anahtar", "istasyon_tip_kodu", "mevsim"}
UYDU = ["s5p_no2_mol_m2", "s5p_so2_mol_m2", "modis_aod550"]


# ---------------------------------------------------------------------------
class Kayit:
    def __init__(self, klasor):
        self.klasor = klasor
        os.makedirs(klasor, exist_ok=True)
        self.log_yol = os.path.join(klasor, "model_log.txt")
        self.durum_yol = os.path.join(klasor, "model_durum.json")

    def log(self, msg):
        s = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(s, flush=True)
        with open(self.log_yol, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    def durum(self, anahtar, deger):
        d = {}
        if os.path.exists(self.durum_yol):
            with open(self.durum_yol, encoding="utf-8") as f:
                d = json.load(f)
        d[anahtar] = {"durum": deger, "zaman": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(self.durum_yol, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------
def veri_yukle(panel_yol, hizli):
    p = pd.read_parquet(panel_yol)
    if hizli:
        p = p[(p["tarih"] >= "2023-01-01") & (p["tarih"] < "2023-07-01")].copy()
    # modelleme oncesi temizlik (hazirlik raporundaki notlar)
    p.loc[p["modis_aod550"] >= 3, "modis_aod550"] = np.nan
    for c in [c for c in p.columns if c.startswith("yapilasma_oran_")]:
        p[c] = p[c].clip(upper=1.0)
    p["doy_sin"] = np.sin(2 * np.pi * p["yilin_gunu"] / 365.25)
    p["doy_cos"] = np.cos(2 * np.pi * p["yilin_gunu"] / 365.25)
    for c in UYDU:
        p[f"{c}_eksik"] = p[c].isna().astype("int8")
    return p


def ozellik_listesi(p):
    oz = []
    for c in p.columns:
        if c in DISLA or c.startswith(DISLA_ON_EK) or c.endswith("_saat"):
            continue
        if not pd.api.types.is_numeric_dtype(p[c]):
            continue
        if p[c].nunique(dropna=True) <= 1:
            continue
        oz.append(c)
    return oz


def katlar(df, sema, n_kat):
    from sklearn.model_selection import GroupKFold, KFold
    kat = np.full(len(df), -1)
    if sema == "rastgele":
        for i, (_, te) in enumerate(KFold(n_kat, shuffle=True, random_state=SEED).split(df)):
            kat[te] = i
    else:
        grup = df["istasyon_id"] if sema == "istasyon" else df["il"]
        # GroupKFold deterministik; gruplari karistirmak icin grup adlarini permute et
        u = grup.unique()
        rng = np.random.RandomState(SEED)
        harita = dict(zip(u, rng.permutation(len(u))))
        g = grup.map(harita).values
        for i, (_, te) in enumerate(GroupKFold(n_kat).split(df, groups=g)):
            kat[te] = i
    return kat


def metrik(y, yhat):
    from sklearn.metrics import r2_score
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    if len(y) < 10:
        return {"n": int(len(y)), "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MB": np.nan, "r": np.nan}
    return {"n": int(len(y)), "R2": r2_score(y, yhat), "RMSE": float(np.sqrt(np.mean((yhat - y) ** 2))),
            "MAE": float(np.mean(np.abs(yhat - y))), "MB": float(np.mean(yhat - y)),
            "r": float(np.corrcoef(y, yhat)[0, 1])}


# ---------------------------------------------------------------------------
class Komsuluk:
    """Istasyon koordinatlari, gunluk deger matrisi ve kat icinde sizintisiz IDW."""

    def __init__(self, df, hedef):
        import geopandas as gpd
        ist = df.groupby("istasyon_id")[["lon", "lat"]].first()
        g = gpd.GeoSeries(gpd.points_from_xy(ist["lon"], ist["lat"]), crs="EPSG:4326").to_crs(LAEA)
        self.ist = ist.index.values
        self.idx = {s: i for i, s in enumerate(self.ist)}
        xy = np.c_[g.x.values, g.y.values] / 1000.0
        self.xy = xy
        self.D = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        self.gunler = np.sort(df["tarih"].unique())
        self.gidx = {d: i for i, d in enumerate(self.gunler)}
        V = np.full((len(self.gunler), len(self.ist)), np.nan, dtype="float64")
        r = df["tarih"].map(self.gidx).values
        c = df["istasyon_id"].map(self.idx).values
        V[r, c] = np.log1p(df[hedef].values)
        self.V = V

    def agirlik(self, egitim_maske, k=IDW_K):
        """egitim_maske: istasyon bazinda bool. Her istasyon icin kendisi haric en yakin k egitim istasyonu."""
        n = len(self.ist)
        W = np.zeros((n, n))
        D = self.D.copy()
        D[:, ~egitim_maske] = np.inf
        np.fill_diagonal(D, np.inf)
        yakin = np.argsort(D, axis=1)[:, :k]
        for i in range(n):
            d = D[i, yakin[i]]
            ok = np.isfinite(d)
            W[i, yakin[i][ok]] = 1.0 / np.maximum(d[ok], 0.5) ** 2
        en_yakin = np.min(D, axis=1)
        return W, en_yakin

    def idw(self, egitim_maske):
        W, en_yakin = self.agirlik(egitim_maske)
        V = self.V
        var = np.isfinite(V)
        num = np.where(var, V, 0.0) @ W.T
        den = var.astype(float) @ W.T
        with np.errstate(invalid="ignore", divide="ignore"):
            I = np.where(den > 0, num / den, np.nan)
        return I, en_yakin

    def satira(self, df, I, en_yakin):
        r = df["tarih"].map(self.gidx).values
        c = df["istasyon_id"].map(self.idx).values
        return I[r, c], en_yakin[c]


# ---------------------------------------------------------------------------
def model_kur(ad, hizli, gpu):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    n = 100 if hizli else None
    if ad == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    if ad == "rf":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestRegressor(n_estimators=n or 150, max_features=0.33, min_samples_leaf=50,
                                                   max_samples=0.2, n_jobs=-1, random_state=SEED))
    if ad == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=n or 800, learning_rate=0.05, max_depth=6, subsample=0.8,
                            colsample_bytree=0.5, min_child_weight=50, reg_lambda=10.0, tree_method="hist",
                            device="cuda" if gpu else "cpu", random_state=SEED, n_jobs=-1)
    if ad in ("lgbm", "lgbm_idw"):
        from lightgbm import LGBMRegressor
        # mekansal genelleme icin duzenlilestirilmis ayar (2 yillik O3/SO2 on testinde istasyon
        # gruplu dogrulamada varsayilan ayardan daha iyi; bkz. model_ayar_notu)
        return LGBMRegressor(n_estimators=n or 1000, learning_rate=0.05, num_leaves=31, min_child_samples=1000,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0,
                             random_state=SEED, n_jobs=-1, verbose=-1)
    raise ValueError(ad)


def ridge_alpha_sec(X, y, grup):
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score
    en_iyi, en_iyi_skor = None, -np.inf
    for a in [1e2, 1e3, 1e4, 1e5]:
        skor = []
        for itr, ite in GroupKFold(3).split(X, y, grup):
            m = model_kur("ridge", True, False).set_params(ridge__alpha=a)
            m.fit(X.iloc[itr], y[itr])
            skor.append(r2_score(y[ite], m.predict(X.iloc[ite])))
        if np.mean(skor) > en_iyi_skor:
            en_iyi, en_iyi_skor = a, np.mean(skor)
    return en_iyi


def tablo_modeli(df, oz, hedef, kat, ad, kom, hizli, gpu, kayit):
    y = np.log1p(df[hedef].values)
    tahmin = np.full(len(df), np.nan)
    for k in range(kat.max() + 1):
        te = kat == k
        tr = ~te
        X = df[oz]
        if ad == "lgbm_idw":
            egitim_ist = set(df.loc[tr, "istasyon_id"])
            maske = np.array([s in egitim_ist for s in kom.ist])
            I, en_yakin = kom.idw(maske)
            idw_s, yak_s = kom.satira(df, I, en_yakin)
            X = X.assign(komsu_idw=idw_s, en_yakin_istasyon_km=yak_s)
        m = model_kur(ad, hizli, gpu)
        if ad == "ridge":
            # ceza katsayisini egitim icinde istasyon gruplu ic dogrulamayla sec (gunluk satir
            # otokorelasyonu nedeniyle satir bazli GCV asiri kucuk ceza seciyor)
            m.set_params(ridge__alpha=ridge_alpha_sec(X[tr], y[tr], df.loc[tr, "istasyon_id"].values))
        m.fit(X[tr], y[tr])
        # log uzayinda egitim araliginin disina tasan tahminleri kirp (dissal tahmin patlamasini onler)
        tahmin[te] = np.clip(m.predict(X[te]), y[tr].min(), y[tr].max())
        kayit.log(f"      kat {k + 1}/{kat.max() + 1} tamam (egitim {tr.sum():,}, test {te.sum():,})")
    return np.expm1(tahmin)


def idw_modeli(df, hedef, kat, kom):
    tahmin = np.full(len(df), np.nan)
    for k in range(kat.max() + 1):
        te = kat == k
        egitim_ist = set(df.loc[~te, "istasyon_id"])
        maske = np.array([s in egitim_ist for s in kom.ist])
        I, en_yakin = kom.idw(maske)
        idw_s, _ = kom.satira(df, I, en_yakin)
        tahmin[te] = idw_s[te]
    return np.expm1(tahmin)


# ---------------------------------------------------------------------------
def gnn_modeli(df, oz, hedef, kat, kom, hizli, kayit):
    import torch
    import torch.nn as nn
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    S = len(kom.ist)
    T = len(kom.gunler)
    statik = [c for c in oz if df.groupby("istasyon_id")[c].nunique(dropna=True).max() <= 1]
    gunluk = [c for c in oz if c not in statik]
    st = df.groupby("istasyon_id")[statik].first().reindex(kom.ist)
    st = ((st - st.mean()) / st.std().replace(0, 1)).fillna(0.0).values.astype("float32")

    r = df["tarih"].map(kom.gidx).values
    c = df["istasyon_id"].map(kom.idx).values
    G = np.zeros((T, S, len(gunluk)), dtype="float32")
    gd = df[gunluk]
    gd = ((gd - gd.mean()) / gd.std().replace(0, 1)).fillna(0.0).values.astype("float32")
    G[r, c] = gd
    Y = kom.V.astype("float32")                       # log1p deger, NaN = yok
    ym, ys = np.nanmean(Y), np.nanstd(Y)
    Yz = (Y - ym) / ys

    # komsuluk: k en yakin, Gauss cekirdegi
    sig = np.median(np.sort(kom.D, axis=1)[:, 1:IDW_K + 1])
    A = np.zeros((S, S), dtype="float32")
    yakin = np.argsort(kom.D, axis=1)[:, 1:IDW_K + 1]
    for i in range(S):
        A[i, yakin[i]] = np.exp(-(kom.D[i, yakin[i]] / sig) ** 2)
    A = np.maximum(A, A.T) + np.eye(S, dtype="float32")
    dinv = 1.0 / np.sqrt(A.sum(1))
    A = (A * dinv[:, None]) * dinv[None, :]

    At = torch.tensor(A, device=cihaz)
    St = torch.tensor(st, device=cihaz)
    Gt = torch.tensor(G, device=cihaz)
    Yt = torch.tensor(np.nan_to_num(Yz, nan=0.0), device=cihaz)
    Mt = torch.tensor(np.isfinite(Yz), device=cihaz)

    class Ag(nn.Module):
        def __init__(self, giris, h=64, L=3, dr=0.2):
            super().__init__()
            self.gir = nn.Sequential(nn.Linear(giris, h), nn.GELU())
            self.katman = nn.ModuleList([nn.Linear(h, h) for _ in range(L)])
            self.norm = nn.ModuleList([nn.LayerNorm(h) for _ in range(L)])
            self.cik = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
            self.dr = nn.Dropout(dr)

        def forward(self, x):                          # x: (B, S, F)
            h = self.dr(self.gir(x))
            for lin, ln in zip(self.katman, self.norm):
                h = h + torch.relu(ln(torch.matmul(At, lin(h))))
            return self.cik(self.dr(h)).squeeze(-1)

    def giris(gun_idx, gorunur, gurultu=0.0):
        B = len(gun_idx)
        x_st = St if gurultu == 0 else St + gurultu * torch.randn_like(St)
        x_st = x_st.unsqueeze(0).expand(B, S, -1)
        x_g = Gt[gun_idx]
        y = Yt[gun_idx] * gorunur
        return torch.cat([x_st, x_g, y.unsqueeze(-1), gorunur.unsqueeze(-1)], dim=-1)

    def tahmin_et(ag, gorunur_m):
        ag.eval()
        out = np.zeros((T, S), dtype="float32")
        with torch.no_grad():
            for b in range(0, T, 64):
                gi = torch.arange(b, min(b + 64, T), device=cihaz)
                gor = (Mt[gi] & gorunur_m.unsqueeze(0)).float()
                out[b:b + len(gi)] = ag(giris(gi, gor)).cpu().numpy()
        return out

    Mn = np.isfinite(Yz)

    def r2_z(pred, dugum):
        mm = Mn[:, dugum]
        y = Yz[:, dugum][mm]
        pv = pred[:, dugum][mm]
        return 1 - ((y - pv) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)

    # Asiri ogrenmeyi onlemek icin (tam veri testinde 5. epoktan sonra test R2 dusuyordu):
    # dropout 0.2, weight decay 1e-2, statik ozelliklere egitimde Gauss gurultusu (0.3),
    # egitim istasyonlarinin %15'i ic dogrulama dugumu, en iyi epok secimi (sabir 4).
    epok = 3 if hizli else 30
    sabir = 4
    batch = 16
    tahmin_tablo = np.full((T, S), np.nan, dtype="float64")
    rng = np.random.RandomState(SEED)
    for k in range(kat.max() + 1):
        test_ist = set(df.loc[kat == k, "istasyon_id"])
        test_np = np.array([s in test_ist for s in kom.ist])
        egitim_idx = np.where(~test_np)[0]
        val_np = np.zeros(S, dtype=bool)
        val_np[rng.choice(egitim_idx, size=max(3, int(0.15 * len(egitim_idx))), replace=False)] = True
        test_m = torch.tensor(test_np, device=cihaz)
        val_m = torch.tensor(val_np, device=cihaz)
        ogren_m = ~test_m & ~val_m
        ag = Ag(st.shape[1] + G.shape[2] + 2).to(cihaz)
        opt = torch.optim.AdamW(ag.parameters(), lr=2e-3, weight_decay=1e-2)
        en_iyi, en_iyi_durum, bekle, en_iyi_e = -np.inf, None, 0, -1
        for e in range(epok):
            ag.train()
            sira = torch.randperm(T, device=cihaz)
            for b in range(0, T, batch):
                gi = sira[b:b + batch]
                gozlem = Mt[gi] & ogren_m.unsqueeze(0)
                maskeli = gozlem & (torch.rand(gozlem.shape, device=cihaz) < 0.3)
                gorunur = (gozlem & ~maskeli).float()
                if maskeli.sum() == 0:
                    continue
                out = ag(giris(gi, gorunur, gurultu=0.3))
                kayip = ((out - Yt[gi]) ** 2)[maskeli].mean()
                opt.zero_grad()
                kayip.backward()
                opt.step()
            skor = r2_z(tahmin_et(ag, ogren_m), val_np)
            if skor > en_iyi:
                en_iyi, en_iyi_e, bekle = skor, e, 0
                en_iyi_durum = {kk: v.detach().clone() for kk, v in ag.state_dict().items()}
            else:
                bekle += 1
                if bekle >= sabir:
                    break
        ag.load_state_dict(en_iyi_durum)
        # test tahmini: tum egitim istasyonlari (dogrulama dugumleri dahil) gorunur
        pred = tahmin_et(ag, ~test_m)
        tahmin_tablo[:, test_np] = pred[:, test_np]
        kayit.log(f"      GNN kat {k + 1}/{kat.max() + 1} tamam (en iyi epok {en_iyi_e + 1}, "
                  f"ic dogrulama R2(log,z) {en_iyi:.3f}, test R2(log,z) {r2_z(pred, test_np):.3f}, cihaz {cihaz})")
    tahmin_log = np.clip(tahmin_tablo * ys + ym, np.nanmin(Y), np.nanmax(Y))
    return np.expm1(tahmin_log[r, c])


# ---------------------------------------------------------------------------
def kriging_ortalama(df, hedef, kat_ist, sema, kayit):
    from pykrige.ok import OrdinaryKriging
    import geopandas as gpd
    o = df.groupby("istasyon_id").agg(y=(hedef, "mean"), n=(hedef, "count"), lon=("lon", "first"),
                                       lat=("lat", "first"), kat=("_kat", "first"))
    g = gpd.GeoSeries(gpd.points_from_xy(o["lon"], o["lat"]), crs="EPSG:4326").to_crs(LAEA)
    o["x"], o["yk"] = g.x.values / 1000, g.y.values / 1000
    o["tahmin"] = np.nan
    for k in sorted(o["kat"].unique()):
        tr, te = o[o["kat"] != k], o[o["kat"] == k]
        try:
            ok = OrdinaryKriging(tr["x"], tr["yk"], np.log1p(tr["y"]), variogram_model="spherical",
                                 nlags=12, enable_plotting=False)
            z, _ = ok.execute("points", te["x"].values, te["yk"].values)
            o.loc[te.index, "tahmin"] = np.expm1(np.asarray(z))
        except Exception as e:
            kayit.log(f"      kriging kat {k} hata: {e}")
    m = metrik(o["y"].values, o["tahmin"].values)
    return m


# ---------------------------------------------------------------------------
def shap_ozeti(df, oz, hedef, kom, cikti, hizli, kayit):
    from lightgbm import LGBMRegressor
    I, en_yakin = kom.idw(np.ones(len(kom.ist), dtype=bool))
    idw_s, yak_s = kom.satira(df, I, en_yakin)
    X = df[oz].assign(komsu_idw=idw_s, en_yakin_istasyon_km=yak_s)
    y = np.log1p(df[hedef].values)
    m = model_kur("lgbm_idw", hizli, False)
    m.fit(X, y)
    rng = np.random.RandomState(SEED)
    orn = rng.choice(len(X), size=min(30000, len(X)), replace=False)
    katki = m.booster_.predict(X.iloc[orn], pred_contrib=True)[:, :-1]
    ozet = pd.DataFrame({"ozellik": X.columns, "ortalama_abs_shap": np.abs(katki).mean(0)})
    ozet.sort_values("ortalama_abs_shap", ascending=False).to_csv(
        os.path.join(cikti, f"{hedef}_shap_onem.csv"), index=False, encoding="utf-8-sig")
    ist = pd.DataFrame(katki, columns=X.columns)
    ist["istasyon_id"] = df["istasyon_id"].iloc[orn].values
    ust = ozet.sort_values("ortalama_abs_shap", ascending=False)["ozellik"].head(15).tolist()
    ist.groupby("istasyon_id")[ust].mean().to_csv(
        os.path.join(cikti, f"{hedef}_shap_istasyon.csv"), encoding="utf-8-sig")
    kayit.log(f"    SHAP tamam: ilk 5 -> {', '.join(ozet.sort_values('ortalama_abs_shap', ascending=False)['ozellik'].head(5))}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kok", default=VARSAYILAN_KOK)
    ap.add_argument("--kirletici", default=",".join(KIRLETICILER))
    ap.add_argument("--sema", default=",".join(SEMALAR))
    ap.add_argument("--model", default=",".join(MODELLER))
    ap.add_argument("--hizli", action="store_true")
    ap.add_argument("--kat", type=int, default=KAT)
    a = ap.parse_args()

    panel = os.path.join(a.kok, "islenmis", "cbs_panel_istasyon_gunluk.parquet")
    cikti = os.path.join(a.kok, "islenmis", "model_test" if a.hizli else "model")
    oof_k = os.path.join(cikti, "oof")
    shap_k = os.path.join(cikti, "shap")
    os.makedirs(oof_k, exist_ok=True)
    os.makedirs(shap_k, exist_ok=True)
    kayit = Kayit(cikti)
    n_kat = 3 if a.hizli else a.kat

    gpu = False
    try:
        import torch
        gpu = torch.cuda.is_available()
        kayit.log(f"torch {torch.__version__}, CUDA: {gpu}" + (f" ({torch.cuda.get_device_name(0)})" if gpu else ""))
    except Exception as e:
        kayit.log(f"torch yuklenemedi: {e}")

    t0 = time.time()
    p = veri_yukle(panel, a.hizli)
    oz = ozellik_listesi(p)
    kayit.log(f"Panel {p.shape}, ozellik sayisi {len(oz)}")
    with open(os.path.join(cikti, "ozellikler.json"), "w", encoding="utf-8") as f:
        json.dump(oz, f, ensure_ascii=False, indent=0)

    kat_satir = []
    krig_satir = []
    for hedef in a.kirletici.split(","):
        df = p[p[hedef].notna()].copy()
        say = df.groupby("istasyon_id")[hedef].transform("count")
        df = df[say >= (20 if a.hizli else MIN_GUN)].reset_index(drop=True)
        kayit.log("=" * 70)
        kayit.log(f"{hedef}: {len(df):,} istasyon-gun, {df['istasyon_id'].nunique()} istasyon")
        kom = Komsuluk(df, hedef)
        for sema in a.sema.split(","):
            kat = katlar(df, sema, n_kat)
            df["_kat"] = kat
            for ad in a.model.split(","):
                if sema == "rastgele" and ad not in RASTGELE_MODELLER:
                    continue
                dosya = os.path.join(oof_k, f"{hedef}_{sema}_{ad}.parquet")
                if os.path.exists(dosya):
                    kayit.log(f"  {hedef} {sema} {ad}: var, atlaniyor")
                    continue
                anahtar = f"{hedef}|{sema}|{ad}"
                kayit.durum(anahtar, "calisiyor")
                kayit.log(f"  {hedef} {sema} {ad} basladi")
                t1 = time.time()
                try:
                    if ad == "idw":
                        yhat = idw_modeli(df, hedef, kat, kom)
                    elif ad == "gnn":
                        yhat = gnn_modeli(df, oz, hedef, kat, kom, a.hizli, kayit)
                    else:
                        yhat = tablo_modeli(df, oz, hedef, kat, ad, kom, a.hizli, gpu, kayit)
                except Exception as e:
                    import traceback
                    kayit.log(f"  HATA {anahtar}: {e}\n{traceback.format_exc()}")
                    kayit.durum(anahtar, f"HATA: {e}")
                    continue
                out = df[["istasyon_id", "il", "tarih", hedef]].rename(columns={hedef: "gozlem"})
                out["tahmin"] = yhat
                out["kat"] = kat
                out.to_parquet(dosya, index=False)
                m = metrik(out["gozlem"].values, out["tahmin"].values)
                kayit.log(f"  {hedef} {sema} {ad}: R2={m['R2']:.3f} RMSE={m['RMSE']:.2f} "
                          f"({(time.time() - t1) / 60:.1f} dk)")
                kayit.durum(anahtar, "tamam")
            if sema != "rastgele":
                km = kriging_ortalama(df, hedef, kat, sema, kayit)
                krig_satir.append({"kirletici": hedef, "sema": sema, **km})
        shap_dosya = os.path.join(shap_k, f"{hedef}_shap_onem.csv")
        if not os.path.exists(shap_dosya) and ("lgbm_idw" in a.model.split(",")):
            shap_ozeti(df, oz, hedef, kom, shap_k, a.hizli, kayit)

    # ----------------------------------------------------------- ozet metrikler
    satir = []
    for f in sorted(os.listdir(oof_k)):
        if not f.endswith(".parquet"):
            continue
        hedef, sema, ad = f[:-8].split("_", 2)
        o = pd.read_parquet(os.path.join(oof_k, f))
        g = metrik(o["gozlem"].values, o["tahmin"].values)
        im = o.groupby("istasyon_id")[["gozlem", "tahmin"]].mean()
        s = metrik(im["gozlem"].values, im["tahmin"].values)
        satir.append({"kirletici": hedef, "sema": sema, "model": ad,
                      **{f"gunluk_{k}": v for k, v in g.items()},
                      **{f"istasyon_ort_{k}": v for k, v in s.items()}})
        for k in sorted(o["kat"].unique()):
            ok_ = o[o["kat"] == k]
            kat_satir.append({"kirletici": hedef, "sema": sema, "model": ad, "kat": int(k),
                              **metrik(ok_["gozlem"].values, ok_["tahmin"].values)})
    ozet = pd.DataFrame(satir)
    ozet.to_csv(os.path.join(cikti, "metrik_ozet.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(kat_satir).to_csv(os.path.join(cikti, "metrik_kat.csv"), index=False, encoding="utf-8-sig")
    if krig_satir:
        kp = os.path.join(cikti, "kriging_ortalama.csv")
        eski = pd.read_csv(kp) if os.path.exists(kp) else pd.DataFrame()
        pd.concat([eski, pd.DataFrame(krig_satir)]).drop_duplicates(["kirletici", "sema"], keep="last").to_csv(
            kp, index=False, encoding="utf-8-sig")
    if len(ozet):
        kayit.log("OZET (gunluk R2 | istasyon ortalamasi R2):")
        for _, s in ozet.sort_values(["kirletici", "sema", "gunluk_R2"], ascending=[True, True, False]).iterrows():
            kayit.log(f"  {s.kirletici:<5} {s.sema:<9} {s.model:<9} gunluk R2={s.gunluk_R2:.3f} "
                      f"RMSE={s.gunluk_RMSE:.2f} | ort R2={s.istasyon_ort_R2:.3f}")
    kayit.durum("TUMU", f"tamam ({(time.time() - t0) / 60:.1f} dk)")
    kayit.log(f"TUMU TAMAM ({(time.time() - t0) / 60:.1f} dk)")


if __name__ == "__main__":
    main()
