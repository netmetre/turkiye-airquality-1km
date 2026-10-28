# -*- coding: utf-8 -*-
"""
CBS makalesi - 1 km izgara degiskenleri, nihai model ve 2025 haritalari
=========================================================================
Girdiler
  veri\\izgara\\        GEE ciktilari (gee_izgara_degiskenleri.js): izg_*.tif, ist_statik_izgara.csv,
                       ist_yillik_izgara.csv, santraller.csv
  veri\\               geoBoundaries ADM1, GHS-POP zip, Gunes atlasi zip, ruzgar atlasi tif, OSM zip, tablo.xls
  islenmis\\cbs_panel_istasyon_gunluk.parquet, islenmis\\model\\ozellikler.json
  cbs_modelleme.py     (ayni klasorde; veri temizligi, IDW ve LightGBM ayarlari buradan alinir)

Asamalar (cikti varsa atlanir; --yeniden HARF ile o asama ve sonrakiler yeniden uretilir)
  I  1 km LAEA izgara, il atamasi, hucre nufusu
  O  OSM yol uzunlugu ve sanayi/maden alani: 100 m raster + dairesel tampon, izgara ve istasyon degerleri
  S  statik ve yillik degiskenlerin izgara/istasyon tablolari (GEE, atlaslar, santraller, TUIK) + uyum raporu
  V  yeni istasyon degiskenleriyle istasyon gruplu 10 katli dogrulama (LightGBM+IDW)
  H  5 uyeli istasyon-bootstrap topluluk, 2025 icin 73 gun tahmin, yillik/mevsimlik harita ve belirsizlik

Tasarim notu: modelin egitildigi istasyon degiskenleri ile haritada kullanilan izgara degiskenleri
AYNI rasterlardan ve AYNI tampon yonteminden uretilir (V asamasi bu degiskenlerle dogrulamayi tekrarlar).

Kullanim
  python cbs_izgara.py                 # tum asamalar
  python cbs_izgara.py --yeniden S     # S, V, H yeniden
  python cbs_izgara.py --hizli         # kisa deneme (az gun, az agac, 2 uye), cikti islenmis\\izgara_test
"""
import argparse
import glob
import json
import os
import sys
import time
import unicodedata
import zipfile

import numpy as np
import pandas as pd

KOK_VARSAYILAN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
LAEA = "+proj=laea +lat_0=39 +lon_0=35 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
HARITA_YILI = 2025
ADIM_GUN = 5
GUN_SAYISI = 73
KIRLETICILER = ["PM10", "NO2", "SO2", "O3"]
UYE_SAYISI = 5
# il bazinda sabit TUIK degiskenleri haritada il siniri basamaklari uretiyor -> harita modelinden cikarildi
TUIK_HARIC = True
SEED = 42
OSM_COZ = 100.0                 # m
OSM_KENAR = 6000.0              # m, en buyuk tampondan fazla
TAMPONLAR = [500, 1000, 3000, 5000]
WC_SINIF = ["agac", "cali", "cayir", "tarim", "yapi", "ciplak", "kar", "su", "sulak"]
YOL_ANA = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"}
YOL_ORTA = {"secondary", "secondary_link", "tertiary", "tertiary_link"}
YOL_HARIC = {"footway", "path", "steps", "cycleway", "bridleway", "pedestrian",
             "track", "track_grade1", "track_grade2", "track_grade3", "track_grade4",
             "track_grade5", "unknown"}
MEVSIM = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
ERA5_BANT = ["temperature_2m", "dewpoint_temperature_2m", "u_component_of_wind_10m",
             "v_component_of_wind_10m", "surface_pressure", "total_precipitation_sum",
             "surface_solar_radiation_downwards_sum"]

# GEE ciktilarindaki bant sirasi (gee_izgara_degiskenleri.js ile ayni)
BANT_TOPO_GHS = ["yukseklik_m", "egim_derece", "yukseklik_ort_1km", "yukseklik_ort_5km",
                 "yukseklik_std_5km", "goreli_yukseklik_5km",
                 "nufus_1000", "yapilasma_oran_1000", "nufus_3000", "yapilasma_oran_3000",
                 "nufus_5000", "yapilasma_oran_5000"]
BANT_WC_KUCUK = [f"wc_{s}_{m}" for m in (500, 1000) for s in WC_SINIF]
BANT_WC_BUYUK = [f"wc_{s}_{m}" for m in (3000, 5000) for s in WC_SINIF]
BANT_YILLIK = [f"{v}_{m}" for m in TAMPONLAR for v in ("ndvi", "gece_isigi")]


# ---------------------------------------------------------------------------
class Yol:
    def __init__(self, kok, hizli):
        self.kok = kok
        self.veri = os.path.join(kok, "veri")
        self.izg_veri = os.path.join(self.veri, "izgara")
        self.cikti = os.path.join(kok, "islenmis", "izgara_test" if hizli else "izgara")
        os.makedirs(self.cikti, exist_ok=True)
        self.panel = os.path.join(kok, "islenmis", "cbs_panel_istasyon_gunluk.parquet")
        self.ozellik = os.path.join(kok, "islenmis", "model", "ozellikler.json")
        self.adm1 = os.path.join(self.veri, "geoBoundaries-TUR-ADM1.geojson")
        self.ghs_pop = os.path.join(self.veri, "GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip")
        self.gunes = os.path.join(self.veri, "Turkey_GISdata_LTAym_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF.zip")
        self.ruzgar = os.path.join(self.veri, "TUR_wind-speed_100m.tif")
        self.osm = os.path.join(self.veri, "turkey-261026-free.shp.zip")
        self.tuik = os.path.join(self.veri, "tablo.xls")

    def c(self, ad):
        return os.path.join(self.cikti, ad)


class Kayit:
    def __init__(self, yol):
        self.dosya = yol.c("izgara_log.txt")
        self.durum_dosya = yol.c("izgara_durum.json")

    def log(self, m):
        s = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
        print(s, flush=True)
        with open(self.dosya, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    def durum(self, k, v):
        d = {}
        if os.path.exists(self.durum_dosya):
            with open(self.durum_dosya, encoding="utf-8") as f:
                d = json.load(f)
        d[k] = {"durum": v, "zaman": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(self.durum_dosya, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)


def il_norm(x):
    x = str(x).strip().replace("İ", "I").replace("ı", "i").upper()
    return "".join(c for c in unicodedata.normalize("NFKD", x) if not unicodedata.combining(c))


def modelleme_modulu(kok):
    # cbs_modelleme.py bu betikle ayni klasorde (kod\)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cbs_modelleme as cm
    return cm


def gunler_2025():
    return pd.date_range(f"{HARITA_YILI}-01-01", periods=GUN_SAYISI, freq=f"{ADIM_GUN}D")


def laea_xy(lon, lat):
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", LAEA, always_xy=True)
    return t.transform(np.asarray(lon, float), np.asarray(lat, float))


def istasyonlar(yol):
    p = pd.read_parquet(yol.panel, columns=["istasyon_id", "il", "lon", "lat"])
    s = p.groupby("istasyon_id").first().reset_index()
    s["x"], s["y"] = laea_xy(s["lon"], s["lat"])
    return s


def izgara_bilgi(yol):
    with open(yol.c("izgara_bilgi.json"), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# I) izgara
# ---------------------------------------------------------------------------
def asama_I(yol, k, a):
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling
    from rasterio.windows import from_bounds
    from pyproj import Transformer

    adm = gpd.read_file(yol.adm1).to_crs(LAEA).reset_index(drop=True)
    adm["il_anahtar"] = adm["shapeName"].map(il_norm)
    minx, miny, maxx, maxy = adm.total_bounds
    x0 = np.floor(minx / 1000) * 1000
    y1 = np.ceil(maxy / 1000) * 1000
    ncol = int(np.ceil((maxx - x0) / 1000))
    nrow = int(np.ceil((y1 - miny) / 1000))
    tr = from_origin(x0, y1, 1000, 1000)
    il_r = rasterize(((g, i + 1) for i, g in enumerate(adm.geometry)), out_shape=(nrow, ncol),
                     transform=tr, fill=0, all_touched=False, dtype="int16")
    satir, sutun = np.nonzero(il_r)
    x = x0 + 500 + sutun * 1000.0
    y = y1 - 500 - satir * 1000.0
    lon, lat = Transformer.from_crs(LAEA, "EPSG:4326", always_xy=True).transform(x, y)
    h = pd.DataFrame({"hucre_id": np.arange(len(x), dtype="int32"), "satir": satir.astype("int32"),
                      "sutun": sutun.astype("int32"), "x": x, "y": y, "lon": lon, "lat": lat,
                      "il_anahtar": adm["il_anahtar"].values[il_r[satir, sutun] - 1]})

    # hucre nufusu (GHS-POP 30 ss -> 1 km LAEA, toplam korunarak)
    if os.path.exists(yol.ghs_pop):
        with zipfile.ZipFile(yol.ghs_pop) as z:
            ad = [n for n in z.namelist() if n.lower().endswith(".tif")][0]
        kaynak = "/vsizip/" + yol.ghs_pop.replace("\\", "/") + "/" + ad
        with rasterio.open(kaynak) as r:
            w = from_bounds(24.5, 34.8, 46.0, 43.2, r.transform).round_offsets().round_lengths()
            arr = r.read(1, window=w).astype("float64")
            nod = r.nodata
            src_tr = r.window_transform(w)
            src_crs = r.crs
        if nod is not None:
            arr[np.isclose(arr, nod)] = 0
        arr[arr < 0] = 0
        hedef = np.zeros((nrow, ncol), dtype="float64")
        reproject(arr, hedef, src_transform=src_tr, src_crs=src_crs, dst_transform=tr,
                  dst_crs=LAEA, resampling=Resampling.sum)
        h["nufus_hucre"] = hedef[satir, sutun]
        k.log(f"  hucre nufusu toplami: {h['nufus_hucre'].sum() / 1e6:.1f} milyon")
    else:
        h["nufus_hucre"] = np.nan
        k.log("  UYARI: GHS-POP zip yok, hucre nufusu bos")
    h.to_parquet(yol.c("I_izgara_hucre.parquet"), index=False)
    with open(yol.c("izgara_bilgi.json"), "w", encoding="utf-8") as f:
        json.dump({"x0": x0, "y1": y1, "nrow": nrow, "ncol": ncol, "crs": LAEA, "hucre": int(len(h))}, f)
    k.log(f"  izgara: {nrow} x {ncol}, Turkiye ici hucre {len(h):,}, il sayisi {h['il_anahtar'].nunique()}")


# ---------------------------------------------------------------------------
# O) OSM
# ---------------------------------------------------------------------------
def disk(r_px, normal):
    yy, xx = np.mgrid[-r_px:r_px + 1, -r_px:r_px + 1]
    d = ((xx ** 2 + yy ** 2) <= r_px ** 2).astype("float32")
    return d / d.sum() if normal else d


def asama_O(yol, k, a):
    from pyogrio import read_dataframe
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from scipy.signal import oaconvolve
    from scipy.ndimage import distance_transform_edt
    import shapely

    b = izgara_bilgi(yol)
    h = pd.read_parquet(yol.c("I_izgara_hucre.parquet"), columns=["hucre_id", "x", "y"])
    st = istasyonlar(yol)
    # 100 m raster: izgara hucre merkezleri piksel merkezine denk gelecek sekilde kaydirilmis
    ox = b["x0"] - OSM_KENAR - OSM_COZ / 2
    oy = b["y1"] + OSM_KENAR + OSM_COZ / 2
    nc = int(round((b["ncol"] * 1000 + 2 * OSM_KENAR) / OSM_COZ)) + 2
    nr = int(round((b["nrow"] * 1000 + 2 * OSM_KENAR) / OSM_COZ)) + 2
    tr = from_origin(ox, oy, OSM_COZ, OSM_COZ)
    k.log(f"  100 m raster {nr} x {nc} ({nr * nc / 1e6:.0f} M piksel)")

    def rc(x, y):
        c = np.floor((np.asarray(x) - ox) / OSM_COZ).astype(int)
        r = np.floor((oy - np.asarray(y)) / OSM_COZ).astype(int)
        return np.clip(r, 0, nr - 1), np.clip(c, 0, nc - 1)

    hr, hc = rc(h["x"], h["y"])
    sr, sc = rc(st["x"], st["y"])
    G = pd.DataFrame({"hucre_id": h["hucre_id"]})
    S = pd.DataFrame({"istasyon_id": st["istasyon_id"]})

    zp = "/vsizip/" + yol.osm.replace("\\", "/")
    k.log("  OSM yollar okunuyor...")
    yolv = read_dataframe(zp + "/gis_osm_roads_free_1.shp", columns=["fclass"])
    yolv = yolv[~yolv["fclass"].isin(YOL_HARIC)].to_crs(LAEA).explode(index_parts=False)
    yolv["sinif"] = np.where(yolv["fclass"].isin(YOL_ANA), "ana",
                             np.where(yolv["fclass"].isin(YOL_ORTA), "orta", "diger"))
    k.log(f"    yol parcasi: {len(yolv):,}")
    toplam = np.zeros(nr * nc, dtype="float32")
    ana_var = None
    for s in ["ana", "orta", "diger"]:
        uz = np.zeros(nr * nc, dtype="float64")
        geo = yolv.loc[yolv["sinif"] == s, "geometry"].values
        for i0 in range(0, len(geo), 200_000):
            seg = shapely.segmentize(geo[i0:i0 + 200_000], 20.0)
            xy, ix = shapely.get_coordinates(seg, return_index=True)
            ayni = ix[1:] == ix[:-1]
            p0, p1 = xy[:-1][ayni], xy[1:][ayni]
            orta = (p0 + p1) / 2
            L = np.hypot(*(p1 - p0).T) / 1000.0
            c = np.floor((orta[:, 0] - ox) / OSM_COZ).astype(np.int64)
            r = np.floor((oy - orta[:, 1]) / OSM_COZ).astype(np.int64)
            ok = (c >= 0) & (c < nc) & (r >= 0) & (r < nr)
            uz += np.bincount(r[ok] * nc + c[ok], weights=L[ok], minlength=nr * nc)
            del seg, xy, ix, p0, p1, orta, L, c, r, ok
        uz = uz.astype("float32")
        toplam += uz
        R = uz.reshape(nr, nc)
        if s == "ana":
            ana_var = R > 0
        for m in TAMPONLAR:
            f = oaconvolve(R, disk(int(round(m / OSM_COZ)), False), mode="same")
            G[f"yol_{s}_km_{m}"] = np.maximum(f[hr, hc], 0)
            S[f"yol_{s}_km_{m}"] = np.maximum(f[sr, sc], 0)
            del f
        k.log(f"    yol {s}: toplam {uz.sum():,.0f} km")
        del uz, R
    R = toplam.reshape(nr, nc)
    for m in TAMPONLAR:
        f = oaconvolve(R, disk(int(round(m / OSM_COZ)), False), mode="same")
        G[f"yol_toplam_km_{m}"] = np.maximum(f[hr, hc], 0)
        S[f"yol_toplam_km_{m}"] = np.maximum(f[sr, sc], 0)
        del f
    del toplam, R, yolv
    d = distance_transform_edt(~ana_var, sampling=OSM_COZ).astype("float32")
    G["uzaklik_ana_yol_m"] = d[hr, hc]
    S["uzaklik_ana_yol_m"] = d[sr, sc]
    del d, ana_var

    k.log("  OSM arazi kullanimi (sanayi, maden ocagi)...")
    lu = read_dataframe(zp + "/gis_osm_landuse_a_free_1.shp", columns=["fclass"])
    lu = lu[lu["fclass"].isin(["industrial", "quarry"])].to_crs(LAEA)
    for fc, ad in [("industrial", "sanayi"), ("quarry", "maden_ocagi")]:
        g = lu.loc[lu["fclass"] == fc, "geometry"]
        R = (rasterize(((x, 1) for x in g), out_shape=(nr, nc), transform=tr, fill=0,
                       all_touched=False, dtype="uint8") if len(g) else np.zeros((nr, nc), "uint8"))
        Rf = R.astype("float32")
        for m in [1000, 3000, 5000]:
            f = oaconvolve(Rf, disk(int(round(m / OSM_COZ)), True), mode="same")
            G[f"{ad}_oran_{m}"] = np.clip(f[hr, hc], 0, 1)
            S[f"{ad}_oran_{m}"] = np.clip(f[sr, sc], 0, 1)
            del f
        if ad == "sanayi":
            d = (distance_transform_edt(R == 0, sampling=OSM_COZ).astype("float32") if R.any()
                 else np.full((nr, nc), np.nan, "float32"))
            G["uzaklik_sanayi_m"] = d[hr, hc]
            S["uzaklik_sanayi_m"] = d[sr, sc]
            del d
        del R, Rf
    G.to_parquet(yol.c("O_osm_izgara.parquet"), index=False)
    S.to_parquet(yol.c("O_osm_istasyon.parquet"), index=False)
    k.log(f"  O tamam: {G.shape[1] - 1} OSM degiskeni")


# ---------------------------------------------------------------------------
# S) statik ve yillik degiskenler
# ---------------------------------------------------------------------------
def gee_oku(yol, onek, beklenen=None):
    """GEE GeoTIFF'ini (gerekirse parcalari birlestirerek) okur: (dizi[b,h,w], transform, adlar)."""
    import rasterio
    from rasterio.merge import merge
    dosyalar = sorted(glob.glob(os.path.join(yol.izg_veri, onek + "*.tif")))
    if not dosyalar:
        raise FileNotFoundError(f"{yol.izg_veri} icinde {onek}*.tif yok")
    if len(dosyalar) > 1 and not all("-0000" in os.path.basename(f) for f in dosyalar):
        raise ValueError(f"{onek}: birden fazla dosya var ve GEE parca adi tasimiyor (eski/yeni kopya?): "
                         f"{[os.path.basename(f) for f in dosyalar]}")
    kaynak = [rasterio.open(f) for f in dosyalar]
    try:
        tanim = kaynak[0].descriptions
        if len(kaynak) == 1:
            arr, trf = kaynak[0].read(), kaynak[0].transform
        else:
            arr, trf = merge(kaynak)
    finally:
        for s in kaynak:
            s.close()
    adlar = list(tanim) if all(tanim) else None
    if beklenen is not None:
        if adlar is not None and adlar != list(beklenen):
            if set(adlar) == set(beklenen):
                arr = arr[[adlar.index(b) for b in beklenen]]
            else:
                raise ValueError(f"{onek}: bant adlari beklenenden farkli: {adlar[:5]}...")
        if arr.shape[0] != len(beklenen):
            raise ValueError(f"{onek}: {arr.shape[0]} bant var, {len(beklenen)} bekleniyordu")
        adlar = list(beklenen)
    return arr, trf, adlar


def nokta_indeks(trf, shape, lon, lat):
    c = np.floor((np.asarray(lon) - trf.c) / trf.a).astype(int)
    r = np.floor((np.asarray(lat) - trf.f) / trf.e).astype(int)
    ok = (r >= 0) & (r < shape[-2]) & (c >= 0) & (c < shape[-1])
    return np.clip(r, 0, shape[-2] - 1), np.clip(c, 0, shape[-1] - 1), ok


def raster_nokta(yol_tif, lon, lat):
    """Buyuk tek bantli rasterdan Turkiye penceresini okuyup noktalarda deger."""
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(yol_tif) as r:
        w = from_bounds(24.5, 34.8, 46.0, 43.2, r.transform).intersection(
            rasterio.windows.Window(0, 0, r.width, r.height)).round_offsets().round_lengths()
        arr = r.read(1, window=w).astype("float64")
        trf = r.window_transform(w)
        nod = r.nodata
    if nod is not None:
        arr[np.isclose(arr, nod)] = np.nan
    rr, cc, ok = nokta_indeks(trf, arr.shape, lon, lat)
    v = arr[rr, cc]
    v[~ok] = np.nan
    return v


def zip_tif(z, ad=None):
    with zipfile.ZipFile(z) as f:
        adlar = [n for n in f.namelist() if n.lower().endswith(".tif")]
    if ad:
        adlar = [n for n in adlar if n.split("/")[-1].upper() == ad.upper()]
    return "/vsizip/" + z.replace("\\", "/") + "/" + adlar[0]


def tuik_oku(yol_tuik):
    """cbs_hazirlik.py ile ayni TUIK okuma ve turetme."""
    t = pd.read_csv(yol_tuik, sep="\t", encoding="cp1254")
    t.columns = [c.strip() for c in t.columns]
    ad = {
        "Toplam nüfus": "tuik_nufus",
        "Nüfus yoğunluğu (kilometrekareye düşen kişi sayısı)": "tuik_nufus_yogunlugu",
        "İl ve ilçe merkezleri nüfusunun toplam nüfus içindeki oranı (%)": "tuik_kentlesme_orani",
        "Otomobil": "tuik_otomobil", "Kamyon": "tuik_kamyon", "Kamyonet": "tuik_kamyonet",
        "Otobüs": "tuik_otobus", "Minibüs": "tuik_minibus",
        "Bin kişi başına otomobil sayısı": "tuik_bin_kisi_otomobil",
        "Otoyol (km)": "tuik_otoyol_km", "İl ve devlet yolu (km)": "tuik_il_devlet_yolu_km",
        "İniş-kalkış yapan uçak sayısı / Toplam": "tuik_ucak_inis_kalkis",
        "Kişi başına GSYH ($)": "tuik_kisi_basi_gsyh_usd",
        "Mesken (MWh)": "tuik_mesken_elektrik_mwh",
        "Kişi başına toplam elektrik tüketimi (KWh)": "tuik_kisi_basi_elektrik_kwh",
        "Kişi başına sanayi elektrik tüketimi (KWh)": "tuik_kisi_basi_sanayi_elektrik_kwh",
        "İmalat": "tuik_girisim_imalat", "Madencilik ve taşocakçılığı": "tuik_girisim_madencilik",
        "Eklekrtik, gaz, buhar ve iklimlendirme üretimi ve dağıtımı": "tuik_girisim_enerji",
    }
    yeni = {}
    for c in t.columns:
        son = c.split(":")[-1].strip() if ":" in c else c
        if son in ad:
            if son == "Toplam" and "girişim" not in c:
                continue
            yeni[c] = ad[son]
        elif "girişim sayıları" in c and son == "Toplam":
            yeni[c] = "tuik_girisim_toplam"
    t = t.rename(columns=yeni).rename(columns={"YIL": "yil", "BÖLGE ADI": "il_tuik"})
    tutulan = ["yil", "il_tuik"] + sorted(set(yeni.values()))
    t = t[tutulan].copy()
    for c in tutulan[2:]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    for c in ["tuik_otoyol_km", "tuik_ucak_inis_kalkis"]:
        t[c] = t[c].fillna(0.0)
    t["il_anahtar"] = t["il_tuik"].map(il_norm)
    t = t.sort_values(["il_anahtar", "yil"])
    deg = [c for c in t.columns if c.startswith("tuik_")]
    t[deg] = t.groupby("il_anahtar")[deg].ffill()
    t["tuik_kisi_basi_mesken_elektrik_kwh"] = t["tuik_mesken_elektrik_mwh"] * 1000 / t["tuik_nufus"]
    t["tuik_bin_kisi_agir_tasit"] = (t[["tuik_kamyon", "tuik_kamyonet", "tuik_otobus", "tuik_minibus"]]
                                     .sum(axis=1, min_count=1) * 1000 / t["tuik_nufus"])
    t["tuik_bin_kisi_imalat_girisim"] = t["tuik_girisim_imalat"] * 1000 / t["tuik_nufus"]
    return t.drop(columns=["il_tuik"])


def santral_degiskenleri(sant, x, y):
    from scipy.spatial import cKDTree
    sx, sy = laea_xy(sant["lon"], sant["lat"])
    cap = sant["capacitymw"].fillna(0).values
    komur = (sant["fuel1"] == "Coal").values
    xy = np.c_[x, y]
    out = {}
    for ad, m in [("fosil", np.ones(len(sant), bool)), ("komur", komur)]:
        t = cKDTree(np.c_[sx[m], sy[m]])
        d, _ = t.query(xy, k=1)
        out[f"uzaklik_{ad}_santral_m"] = np.minimum(d, 300000.0)
    for ad, m, r in [("fosil_kapasite_mw_10km", np.ones(len(sant), bool), 10000),
                     ("fosil_kapasite_mw_25km", np.ones(len(sant), bool), 25000),
                     ("komur_kapasite_mw_25km", komur, 25000)]:
        t = cKDTree(np.c_[sx[m], sy[m]])
        c = cap[m]
        komsu = t.query_ball_point(xy, r=r)
        out[ad] = np.array([c[i].sum() if len(i) else 0.0 for i in komsu])
    return pd.DataFrame(out)


def asama_S(yol, k, a):
    h = pd.read_parquet(yol.c("I_izgara_hucre.parquet"))
    st = istasyonlar(yol)

    # GEE statik rasterlari -> izgara
    G = h[["hucre_id", "il_anahtar", "lon", "lat"]].copy()
    for onek, bant in [("izg_statik_topo_ghs", BANT_TOPO_GHS), ("izg_statik_wc_kucuk", BANT_WC_KUCUK),
                       ("izg_statik_wc_buyuk", BANT_WC_BUYUK), (f"izg_yillik_{HARITA_YILI}", BANT_YILLIK)]:
        arr, trf, adlar = gee_oku(yol, onek, bant)
        rr, cc, ok = nokta_indeks(trf, arr.shape, h["lon"], h["lat"])
        for i, b in enumerate(adlar):
            v = arr[i, rr, cc].astype("float64")
            v[~ok] = np.nan
            G[b] = v
        k.log(f"  {onek}: {len(adlar)} bant, izgara eksik orani {np.mean(np.isnan(G[adlar[0]])):.3f}")
        del arr

    # yerel nokta rasterlari (istasyon degerleriyle ayni yontem)
    G["nufus_piksel"] = raster_nokta(zip_tif(yol.ghs_pop), h["lon"], h["lat"]) if os.path.exists(yol.ghs_pop) else np.nan
    G["gsa_ghi_kwh_m2_gun"] = raster_nokta(zip_tif(yol.gunes, "GHI.tif"), h["lon"], h["lat"])
    G["gsa_sicaklik_uzun_donem_c"] = raster_nokta(zip_tif(yol.gunes, "TEMP.tif"), h["lon"], h["lat"])
    G["gwa_ruzgar_100m_ms"] = raster_nokta(yol.ruzgar, h["lon"], h["lat"])

    # OSM ve santraller
    G = G.merge(pd.read_parquet(yol.c("O_osm_izgara.parquet")), on="hucre_id", how="left")
    sant = pd.read_csv(os.path.join(yol.izg_veri, "santraller.csv"))
    k.log(f"  santral sayisi (bolge): {len(sant)}, Turkiye: {(sant['country'] == 'TUR').sum()}")
    G = pd.concat([G.reset_index(drop=True), santral_degiskenleri(sant, h["x"].values, h["y"].values)], axis=1)

    # TUIK (harita yili)
    tu = tuik_oku(yol.tuik)
    tu = tu[tu["yil"] == HARITA_YILI].drop(columns=["yil"])
    G = G.merge(tu, on="il_anahtar", how="left")
    k.log(f"  TUIK eslesmeyen hucre: {int(G['tuik_nufus'].isna().sum())}")
    for c in [c for c in G.columns if c.startswith("yapilasma_oran_")]:
        G[c] = G[c].clip(upper=1.0)
    G.to_parquet(yol.c("S_izgara_statik.parquet"), index=False)

    # istasyonlarin yeni statik degerleri (ayni rasterlar)
    gs = pd.read_csv(os.path.join(yol.izg_veri, "ist_statik_izgara.csv"))
    gs = gs[["istasyon_id"] + [c for c in BANT_TOPO_GHS + BANT_WC_KUCUK + BANT_WC_BUYUK if c in gs.columns]]
    S = st[["istasyon_id"]].merge(gs, on="istasyon_id", how="left")
    S = S.merge(pd.read_parquet(yol.c("O_osm_istasyon.parquet")), on="istasyon_id", how="left")
    S = pd.concat([S.reset_index(drop=True), santral_degiskenleri(sant, st["x"].values, st["y"].values)], axis=1)
    for c in [c for c in S.columns if c.startswith("yapilasma_oran_")]:
        S[c] = S[c].clip(upper=1.0)
    S.to_parquet(yol.c("S_istasyon_statik.parquet"), index=False)

    gy = pd.read_csv(os.path.join(yol.izg_veri, "ist_yillik_izgara.csv"))
    gy = gy[["istasyon_id", "yil"] + [c for c in BANT_YILLIK if c in gy.columns]]
    son = gy[gy["yil"] == gy["yil"].max()].copy()
    son["yil"] = 2026
    gy = pd.concat([gy, son], ignore_index=True)
    gy.to_parquet(yol.c("S_istasyon_yillik.parquet"), index=False)

    # uyum raporu: eski istasyon degerleri ile yeni (izgara yontemi) degerler
    p = pd.read_parquet(yol.panel)
    eski = p.groupby("istasyon_id").first()
    eski_y = p.groupby(["istasyon_id", "yil"]).first()
    sat = []
    for c in [c for c in S.columns if c != "istasyon_id" and c in eski.columns]:
        e = S.set_index("istasyon_id")[c].reindex(eski.index)
        sat.append((c, eski[c], e))
    gyi = gy.set_index(["istasyon_id", "yil"])
    for c in [c for c in BANT_YILLIK if c in eski_y.columns]:
        sat.append((c, eski_y[c], gyi[c].reindex(eski_y.index)))
    r = []
    for c, e, y in sat:
        m = e.notna() & y.notna()
        if m.sum() < 10:
            r.append({"degisken": c, "n": int(m.sum())})
            continue
        r.append({"degisken": c, "n": int(m.sum()), "pearson": np.corrcoef(e[m], y[m])[0, 1],
                  "spearman": e[m].rank().corr(y[m].rank()), "eski_ort": e[m].mean(), "yeni_ort": y[m].mean(),
                  "eski_medyan": e[m].median(), "yeni_medyan": y[m].median()})
    r = pd.DataFrame(r).sort_values("spearman")
    r.to_csv(yol.c("S_uyum_raporu.csv"), index=False, encoding="utf-8-sig")
    k.log(f"  uyum raporu: {len(r)} degisken, Spearman medyan {r['spearman'].median():.3f}, "
          f"en dusuk: {', '.join(f'{a}={b:.2f}' for a, b in r[['degisken', 'spearman']].head(5).values)}")
    k.log(f"  S tamam: izgara {G.shape}, istasyon {S.shape}")


# ---------------------------------------------------------------------------
# Ortak: yeni degiskenlerle egitim paneli
# ---------------------------------------------------------------------------
def egitim_paneli(yol, cm):
    p = cm.veri_yukle(yol.panel, False)
    S = pd.read_parquet(yol.c("S_istasyon_statik.parquet"))
    Y = pd.read_parquet(yol.c("S_istasyon_yillik.parquet"))
    deg_s = [c for c in S.columns if c != "istasyon_id" and c in p.columns]
    deg_y = [c for c in Y.columns if c not in ("istasyon_id", "yil") and c in p.columns]
    p = p.drop(columns=deg_s + deg_y)
    p = p.merge(S[["istasyon_id"] + deg_s], on="istasyon_id", how="left")
    p = p.merge(Y[["istasyon_id", "yil"] + deg_y], on=["istasyon_id", "yil"], how="left")
    for c in [c for c in p.columns if c.startswith("yapilasma_oran_")]:
        p[c] = p[c].clip(upper=1.0)
    with open(yol.ozellik, encoding="utf-8") as f:
        oz = json.load(f)
    eksik = [c for c in oz if c not in p.columns]
    if eksik:
        raise ValueError(f"egitim panelinde eksik ozellik: {eksik}")
    if TUIK_HARIC:
        oz = [c for c in oz if not c.startswith("tuik_")]
    return p, oz, deg_s + deg_y


def hedef_verisi(p, hedef, cm, hizli):
    df = p[p[hedef].notna()].copy()
    say = df.groupby("istasyon_id")[hedef].transform("count")
    df = df[say >= (20 if hizli else cm.MIN_GUN)].reset_index(drop=True)
    return df


class SessizKayit:
    def __init__(self, k):
        self.k = k

    def log(self, m):
        if "tamam" not in m:
            self.k.log(m)


# ---------------------------------------------------------------------------
# V) dogrulama (yeni degiskenlerle, istasyon gruplu)
# ---------------------------------------------------------------------------
def asama_V(yol, k, a):
    cm = modelleme_modulu(yol.kok)
    p, oz, degisen = egitim_paneli(yol, cm)
    if a.hizli:
        p = p[(p["tarih"] >= "2023-01-01") & (p["tarih"] < "2023-07-01")].copy()
    k.log(f"  egitim paneli {p.shape}, {len(oz)} ozellik, yeniden uretilen {len(degisen)} degisken")
    eski = pd.read_csv(os.path.join(yol.kok, "islenmis", "model", "metrik_ozet.csv"), encoding="utf-8-sig")
    sat = []
    for hedef in a.kirletici:
        df = hedef_verisi(p, hedef, cm, a.hizli)
        kom = cm.Komsuluk(df, hedef)
        kat = cm.katlar(df, "istasyon", 3 if a.hizli else cm.KAT)
        t0 = time.time()
        yhat = cm.tablo_modeli(df, oz, hedef, kat, "lgbm_idw", kom, a.hizli, False, SessizKayit(k))
        y = df[hedef].values
        g = cm.metrik(y, yhat)
        o = pd.DataFrame({"i": df["istasyon_id"], "y": y, "p": yhat}).groupby("i").mean()
        go = cm.metrik(o["y"].values, o["p"].values)
        # log1p geri donusum sapmasi: dis-kat tahminlerinden ortalama orani (carpimsal duzeltme)
        ok = np.isfinite(y) & np.isfinite(yhat)
        kats = float(y[ok].mean() / yhat[ok].mean())
        gd = cm.metrik(y, yhat * kats)
        od = cm.metrik(o["y"].values, o["p"].values * kats)
        e = eski.query("kirletici==@hedef and sema=='istasyon' and model=='lgbm_idw'")
        sat.append({"kirletici": hedef, "gunluk_R2_yeni": g["R2"], "gunluk_RMSE_yeni": g["RMSE"],
                    "istasyon_ort_R2_yeni": go["R2"], "duzeltme_katsayisi": kats,
                    "ort_sapma_once": g["MB"], "ort_sapma_sonra": gd["MB"],
                    "gunluk_R2_duz": gd["R2"], "gunluk_RMSE_duz": gd["RMSE"], "istasyon_ort_R2_duz": od["R2"],
                    "gunluk_R2_eski": e["gunluk_R2"].iloc[0] if len(e) else np.nan,
                    "istasyon_ort_R2_eski": e["istasyon_ort_R2"].iloc[0] if len(e) else np.nan})
        k.log(f"  {hedef}: gunluk R2 {g['R2']:.3f} (eski {sat[-1]['gunluk_R2_eski']:.3f}), istasyon ort R2 "
              f"{go['R2']:.3f} (eski {sat[-1]['istasyon_ort_R2_eski']:.3f}); duzeltme katsayisi {kats:.3f}, "
              f"sapma {g['MB']:.2f} -> {gd['MB']:.2f}, duzeltilmis gunluk R2 {gd['R2']:.3f}, "
              f"ist ort R2 {od['R2']:.3f} [{(time.time() - t0) / 60:.1f} dk]")
    pd.DataFrame(sat).to_csv(yol.c("V_dogrulama.csv"), index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# H) topluluk modeli ve haritalar
# ---------------------------------------------------------------------------
class GunlukIzgara:
    """2025 gunluk uydu ve ERA5 degiskenlerini izgara hucrelerinde verir."""

    def __init__(self, yol, h, k):
        from scipy.spatial import cKDTree
        self.n = len(h)
        gun_adlari = [f"g{i:03d}" for i in range(GUN_SAYISI)]
        self.no2, trf, _ = gee_oku(yol, f"izg_s5p_no2_{HARITA_YILI}", gun_adlari)
        self.so2, _, _ = gee_oku(yol, f"izg_s5p_so2_{HARITA_YILI}", gun_adlari)
        self.aod, _, _ = gee_oku(yol, f"izg_aod_{HARITA_YILI}", gun_adlari)
        self.r, self.c, self.ok = nokta_indeks(trf, self.no2.shape, h["lon"], h["lat"])
        era_adlar = [f"{i}_{b}" for i in range(GUN_SAYISI) for b in ERA5_BANT]
        self.era, etr, _ = gee_oku(yol, f"izg_era5_{HARITA_YILI}")
        if self.era.shape[0] != len(era_adlar):
            raise ValueError(f"ERA5 bant sayisi {self.era.shape[0]}, beklenen {len(era_adlar)}")
        self.era = self.era.reshape(GUN_SAYISI, len(ERA5_BANT), *self.era.shape[1:]).astype("float32")
        er, ec, _ = nokta_indeks(etr, self.era.shape, h["lon"], h["lat"])
        # kiyidaki bos ERA5 pikselleri: en yakin dolu piksel
        gecerli = np.isfinite(self.era[0, 0]) & (self.era[0, 0] != 0)
        vr, vc = np.nonzero(gecerli)
        t = cKDTree(np.c_[vr, vc])
        _, j = t.query(np.c_[er, ec], k=1)
        bos = ~gecerli[er, ec]
        er[bos], ec[bos] = vr[j[bos]], vc[j[bos]]
        self.er, self.ec = er, ec
        k.log(f"  gunluk izgara verisi: ERA5 kiyi doldurulan hucre {int(bos.sum()):,}")

    def gun(self, i):
        def oku(dizi, carpan):
            v = dizi[i, self.r, self.c].astype("float64")
            v[(v == -32768) | ~self.ok] = np.nan
            return v * carpan
        d = {"s5p_no2_mol_m2": oku(self.no2, 1e-7), "s5p_so2_mol_m2": oku(self.so2, 1e-6),
             "modis_aod550": oku(self.aod, 0.001)}
        d["modis_aod550"][d["modis_aod550"] >= 3] = np.nan
        e = self.era[i][:, self.er, self.ec].astype("float64")
        t = e[0] - 273.15
        td = e[1] - 273.15
        d.update({
            "era5_sicaklik_c": t, "era5_cig_noktasi_c": td,
            "era5_bagil_nem_pct": 100 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t)),
            "era5_ruzgar_hiz_ms": np.hypot(e[2], e[3]), "era5_ruzgar_u_ms": e[2], "era5_ruzgar_v_ms": e[3],
            "era5_basinc_hpa": e[4] / 100.0, "era5_yagis_mm": e[5] * 1000.0, "era5_gunes_mj_m2": e[6] / 1e6})
        for c in ["s5p_no2_mol_m2", "s5p_so2_mol_m2", "modis_aod550"]:
            d[f"{c}_eksik"] = np.isnan(d[c]).astype("float64")
        return d


def tif_yaz(dosya, b, h, deger):
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS
    arr = np.full((b["nrow"], b["ncol"]), -9999.0, dtype="float32")
    arr[h["satir"].values, h["sutun"].values] = np.where(np.isfinite(deger), deger, -9999.0)
    with rasterio.open(dosya, "w", driver="GTiff", height=b["nrow"], width=b["ncol"], count=1,
                       dtype="float32", crs=CRS.from_proj4(LAEA), nodata=-9999.0,
                       transform=from_origin(b["x0"], b["y1"], 1000, 1000), compress="deflate") as r:
        r.write(arr, 1)


def asama_H(yol, k, a):
    from scipy.spatial import cKDTree
    cm = modelleme_modulu(yol.kok)
    b = izgara_bilgi(yol)
    h = pd.read_parquet(yol.c("I_izgara_hucre.parquet"))
    G = pd.read_parquet(yol.c("S_izgara_statik.parquet"))
    assert (G["hucre_id"].values == h["hucre_id"].values).all()
    p, oz, _ = egitim_paneli(yol, cm)
    # bellek: yalniz gerekli sutunlar, ozellikler float32
    p = p[["istasyon_id", "tarih", "lon", "lat"] + [c for c in KIRLETICILER if c not in oz]
          + [c for c in oz if c not in ("lon", "lat")]].copy()
    for c in oz:
        if c not in ("lon", "lat"):
            p[c] = p[c].astype("float32")
    gunler = gunler_2025()
    gun_i = list(range(0, GUN_SAYISI, 18)) if a.hizli else list(range(GUN_SAYISI))
    gi = GunlukIzgara(yol, h, k)
    gunluk_ad = set(gi.gun(0).keys())
    tak_ad = {"yil", "ay", "yilin_gunu", "haftanin_gunu", "doy_sin", "doy_cos"}
    statik_ad = [c for c in oz if c not in gunluk_ad and c not in tak_ad]
    eksik = [c for c in statik_ad if c not in G.columns]
    if eksik:
        raise ValueError(f"izgara statik tablosunda eksik: {eksik}")
    X_stat = G[statik_ad].to_numpy("float32")
    del G
    poz = {c: i for i, c in enumerate(oz)}
    n_uye = 2 if a.hizli else UYE_SAYISI
    sonuc = h[["hucre_id", "il_anahtar", "lon", "lat", "nufus_hucre"]].copy()
    gxy = np.c_[h["x"].values, h["y"].values] / 1000.0
    vd = pd.read_csv(yol.c("V_dogrulama.csv"), encoding="utf-8-sig").set_index("kirletici")
    k.log(f"  ozellik sayisi {len(oz)} (TUIK haric: {TUIK_HARIC})")

    for hedef in a.kirletici:
        t0 = time.time()
        kats = float(vd.loc[hedef, "duzeltme_katsayisi"]) if hedef in vd.index else 1.0
        k.log(f"  {hedef}: sapma duzeltme katsayisi {kats:.3f}")
        df = hedef_verisi(p, hedef, cm, a.hizli)
        kom = cm.Komsuluk(df, hedef)
        gun_harita = {pd.Timestamp(d): i for d, i in kom.gidx.items()}
        I, en_yakin = kom.idw(np.ones(len(kom.ist), dtype=bool))
        idw_s, yak_s = kom.satira(df, I, en_yakin)
        Xtr = df[oz].assign(komsu_idw=idw_s, en_yakin_istasyon_km=yak_s)
        ytr = np.log1p(df[hedef].values)
        # izgara icin sabit k en yakin istasyon agirliklari
        d, j = cKDTree(kom.xy).query(gxy, k=cm.IDW_K)
        W = 1.0 / np.maximum(d, 0.5) ** 2
        yakin_km = d[:, 0]
        # istasyon bootstrap uyeleri
        rng = np.random.RandomState(SEED)
        ist = df["istasyon_id"].unique()
        satirlar = df.groupby("istasyon_id").indices
        uyeler = []
        for u in range(n_uye):
            sec = rng.choice(ist, size=len(ist), replace=True)
            ix = np.concatenate([satirlar[s] for s in sec])
            m = cm.model_kur("lgbm_idw", a.hizli, False)
            m.fit(Xtr.iloc[ix], ytr[ix])
            uyeler.append((m, ytr[ix].min(), ytr[ix].max()))
            del ix
        n_satir = len(df)
        del Xtr, df, idw_s, yak_s, I
        k.log(f"  {hedef}: {n_satir:,} satir, {len(ist)} istasyon, {n_uye} uye egitildi "
              f"[{(time.time() - t0) / 60:.1f} dk]")

        top = np.zeros((n_uye, len(h)))
        say = np.zeros((n_uye, len(h)))
        mev_top = {s: np.zeros(len(h)) for s in ("DJF", "MAM", "JJA", "SON")}
        mev_say = {s: np.zeros(len(h)) for s in ("DJF", "MAM", "JJA", "SON")}
        X = np.empty((len(h), len(oz) + 2), dtype="float32")
        X[:, [poz[c] for c in statik_ad]] = X_stat
        for n, i in enumerate(gun_i):
            gun = gunler[i]
            dd = gi.gun(i)
            for c, v in dd.items():
                if c in poz:
                    X[:, poz[c]] = v
            tak = {"yil": gun.year, "ay": gun.month, "yilin_gunu": gun.dayofyear, "haftanin_gunu": gun.dayofweek,
                   "doy_sin": np.sin(2 * np.pi * gun.dayofyear / 365.25),
                   "doy_cos": np.cos(2 * np.pi * gun.dayofyear / 365.25)}
            for c, v in tak.items():
                if c in poz:
                    X[:, poz[c]] = v
            r = gun_harita.get(pd.Timestamp(gun))
            if r is None:
                X[:, len(oz)] = np.nan
            else:
                v = kom.V[r][j]
                var = np.isfinite(v)
                den = (W * var).sum(1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    X[:, len(oz)] = np.where(den > 0, (W * np.where(var, v, 0)).sum(1) / den, np.nan)
            X[:, len(oz) + 1] = yakin_km
            gun_ort = np.zeros(len(h))
            for u, (m, lo, hi) in enumerate(uyeler):
                tah = np.expm1(np.clip(m.predict(X), lo, hi)) * kats
                top[u] += tah
                say[u] += 1
                gun_ort += tah / n_uye
            s = MEVSIM[gun.month]
            mev_top[s] += gun_ort
            mev_say[s] += 1
            if (n + 1) % 10 == 0 or n == len(gun_i) - 1:
                k.log(f"    {hedef} gun {n + 1}/{len(gun_i)} [{(time.time() - t0) / 60:.1f} dk]")
        uye_ort = top / say
        ort = uye_ort.mean(0)
        sd = uye_ort.std(0, ddof=1) if n_uye > 1 else np.full(len(h), np.nan)
        sonuc[f"{hedef}_yillik"] = ort
        sonuc[f"{hedef}_sd"] = sd
        tif_yaz(yol.c(f"{hedef}_{HARITA_YILI}_yillik.tif"), b, h, ort)
        tif_yaz(yol.c(f"{hedef}_{HARITA_YILI}_sd.tif"), b, h, sd)
        for s in mev_top:
            v = np.where(mev_say[s] > 0, mev_top[s] / np.maximum(mev_say[s], 1), np.nan)
            sonuc[f"{hedef}_{s}"] = v
            tif_yaz(yol.c(f"{hedef}_{HARITA_YILI}_{s}.tif"), b, h, v)
        k.log(f"  {hedef} harita tamam: ortalama {np.nanmean(ort):.1f}, medyan {np.nanmedian(ort):.1f}, "
              f"%1-%99 {np.nanpercentile(ort, 1):.1f}-{np.nanpercentile(ort, 99):.1f}, "
              f"ortalama SD {np.nanmean(sd):.2f} [{(time.time() - t0) / 60:.1f} dk]")
        sonuc.to_parquet(yol.c(f"H_harita_{HARITA_YILI}.parquet"), index=False)


# ---------------------------------------------------------------------------
ASAMALAR = [("I", asama_I, "I_izgara_hucre.parquet"), ("O", asama_O, "O_osm_izgara.parquet"),
            ("S", asama_S, "S_izgara_statik.parquet"), ("V", asama_V, "V_dogrulama.csv"),
            ("H", asama_H, f"H_harita_{HARITA_YILI}.parquet")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kok", default=KOK_VARSAYILAN)
    ap.add_argument("--yeniden", default="")
    ap.add_argument("--hizli", action="store_true")
    ap.add_argument("--kirletici", default=",".join(KIRLETICILER))
    ap.add_argument("--asama", default="", help="yalniz bu asamalari calistir, or. S,V")
    a = ap.parse_args()
    a.kirletici = a.kirletici.split(",")
    yol = Yol(a.kok, a.hizli)
    k = Kayit(yol)
    harfler = [x for x, _, _ in ASAMALAR]
    if a.yeniden:
        for x, _, dosya in ASAMALAR[harfler.index(a.yeniden.upper()):]:
            if os.path.exists(yol.c(dosya)):
                os.remove(yol.c(dosya))
                k.log(f"{x} ciktisi silindi (yeniden uretilecek)")
    secili = set(a.asama.upper().split(",")) if a.asama else set(harfler)
    t0 = time.time()
    for x, f, dosya in ASAMALAR:
        if x not in secili:
            continue
        if os.path.exists(yol.c(dosya)):
            k.log(f"ASAMA {x}: cikti var, atlaniyor")
            continue
        k.log("=" * 70)
        k.log(f"ASAMA {x} basladi")
        k.durum(x, "calisiyor")
        t1 = time.time()
        try:
            f(yol, k, a)
            k.durum(x, f"tamam ({(time.time() - t1) / 60:.1f} dk)")
            k.log(f"ASAMA {x} tamam ({(time.time() - t1) / 60:.1f} dk)")
        except Exception as e:
            import traceback
            k.durum(x, f"HATA: {e}")
            k.log(f"ASAMA {x} HATA: {e}\n{traceback.format_exc()}")
            sys.exit(1)
    k.durum("TUMU", f"tamam ({(time.time() - t0) / 60:.1f} dk)")
    k.log(f"TUMU TAMAM ({(time.time() - t0) / 60:.1f} dk)")


if __name__ == "__main__":
    main()
