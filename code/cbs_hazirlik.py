# -*- coding: utf-8 -*-
"""
CBS makalesi - veri hazirlik betigi
====================================
Cikti: istasyon x gun paneli (356 SIM istasyonu, 2019-01-01 .. 2026-07-27)

Asamalar (her biri ciktisini kaydeder; cikti varsa asama atlanir):
  A) sondosya.csv -> istasyon gunluk ortalamalari (saatlik QC + %75 doluluk)
  B) istasyon statik degiskenleri (WorldCover, GHS-POP/BUILT, gunes, ruzgar,
     OSM yol ve sanayi alani, topografya, santral)
  C) yillik degiskenler (NDVI, gece isigi, TUIK)
  D) gunluk panel: A + Sentinel-5P + MODIS AOD + ERA5-Land + B + C
  E) kontrol raporu ve veri sozlugu

Kullanim:
  python cbs_hazirlik.py            # tum asamalar
  python cbs_hazirlik.py --yeniden B  # sadece B'yi silip yeniden uret (sonraki asamalar da yenilenir)
"""
import argparse
import json
import os
import re
import sys
import time
import unicodedata
import zipfile

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# YOLLAR
# ----------------------------------------------------------------------------
CBS_KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
KOK = os.environ.get("CBS_UST_KLASOR", os.path.dirname(CBS_KOK))  # folder that holds the raw station data folder
CBS = CBS_KOK
VERI = os.path.join(CBS, "veri")
CIKTI = os.path.join(CBS, "islenmis")
CALISAN = os.path.join(KOK, "meteorolojik ve çevresel veriler", "cevresel veriler", "calisan")

SONDOSYA = os.path.join(CALISAN, "sondosya.csv")
ISTASYON_LISTESI = os.path.join(CALISAN, "station_list.csv")

WORLDCOVER_KLASOR = os.path.join(VERI, "worldcover")
GHS_POP_ZIP = os.path.join(VERI, "GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip")
GHS_BUILT_ZIP = os.path.join(VERI, "GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip")
GUNES_ZIP = os.path.join(VERI, "Turkey_GISdata_LTAym_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF.zip")
RUZGAR_TIF = os.path.join(VERI, "TUR_wind-speed_100m.tif")
OSM_ZIP = os.path.join(VERI, "turkey-261026-free.shp.zip")
TUIK = os.path.join(VERI, "tablo.xls")
S5P = os.path.join(VERI, "s5p_no2_so2_gunluk.csv")
AOD = os.path.join(VERI, "modis_aod_gunluk_v2.csv")
ERA5 = os.path.join(VERI, "era5land_gunluk_v2.csv")
TOPO = os.path.join(VERI, "topografya_statik.csv")
SANTRAL = os.path.join(VERI, "santral_yakinlik.csv")
NDVI = os.path.join(VERI, "ndvi_gecesi_yillik.csv")

A_CIKTI = os.path.join(CIKTI, "A_istasyon_gunluk_kirletici.parquet")
B_CIKTI = os.path.join(CIKTI, "B_istasyon_statik.parquet")
C_CIKTI = os.path.join(CIKTI, "C_istasyon_yillik.parquet")
D_CIKTI = os.path.join(CIKTI, "cbs_panel_istasyon_gunluk.parquet")
RAPOR = os.path.join(CIKTI, "hazirlik_raporu.txt")
SOZLUK = os.path.join(CIKTI, "veri_sozlugu.csv")
LOG = os.path.join(CIKTI, "hazirlik_log.txt")
DURUM = os.path.join(CIKTI, "hazirlik_durum.json")

# ----------------------------------------------------------------------------
# PARAMETRELER
# ----------------------------------------------------------------------------
BASLANGIC = pd.Timestamp("2019-01-01")
BITIS = pd.Timestamp("2026-07-28")          # dahil degil
CHUNK = 2_000_000
MIN_SAAT = 18                                # gunluk ortalama icin en az 18/24 saat (%75)
POLLUTANTS = ["PM10", "PM25", "SO2", "CO", "NO2", "NOX", "NO", "O3"]
# EAAI-26-28429 ile ayni QC esikleri (veri_hazirlama_pipeline_DUZELTILMIS.py)
BILINEN_KODLAR = [-9999, -999, -1, 9999, -9999.99]
UST_ESIK = {"PM10": 954.99, "PM25": 575.44, "SO2": 1380.38, "CO": 11481.54,
            "NO2": 524.81, "NOX": 1445.44, "NO": 1000.00, "O3": 630.96}
TAMPONLAR = [500, 1000, 3000, 5000]          # metre
# Turkiye merkezli Lambert esit alan projeksiyonu (uzunluk/alan hesaplari icin)
LAEA = "+proj=laea +lat_0=39 +lon_0=35 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
WC_SINIF = {10: "agac", 20: "cali", 30: "cayir", 40: "tarim", 50: "yapi",
            60: "ciplak", 70: "kar", 80: "su", 90: "sulak"}
YOL_ANA = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"}
YOL_ORTA = {"secondary", "secondary_link", "tertiary", "tertiary_link"}
YOL_HARIC = {"footway", "path", "steps", "cycleway", "bridleway", "pedestrian",
             "track", "track_grade1", "track_grade2", "track_grade3", "track_grade4",
             "track_grade5", "unknown"}

os.makedirs(CIKTI, exist_ok=True)


def log(msg):
    satir = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(satir, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(satir + "\n")


def durum_yaz(asama, deger):
    d = {}
    if os.path.exists(DURUM):
        with open(DURUM, encoding="utf-8") as f:
            d = json.load(f)
    d[asama] = {"durum": deger, "zaman": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(DURUM, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)


def il_norm(x):
    x = str(x).strip().replace("İ", "I").replace("ı", "i").upper()
    return "".join(c for c in unicodedata.normalize("NFKD", x) if not unicodedata.combining(c))


def istasyonlari_oku():
    s = pd.read_csv(ISTASYON_LISTESI)
    xy = s["Location"].str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(float)
    s["lon"], s["lat"] = xy[0], xy[1]
    s = s.rename(columns={"StationId": "istasyon_id", "StationName": "istasyon",
                          "CityName": "il", "StationSubType": "istasyon_tip_kodu"})
    s["istasyon"] = s["istasyon"].astype(str).str.strip()
    s["il_anahtar"] = s["il"].map(il_norm)
    return s[["istasyon_id", "istasyon", "il", "il_anahtar", "istasyon_tip_kodu", "lon", "lat"]]


# ----------------------------------------------------------------------------
# A) sondosya.csv -> istasyon gunluk
# ----------------------------------------------------------------------------
def asama_A():
    log("=" * 70)
    log("ASAMA A: sondosya.csv -> istasyon gunluk ortalamalari")
    parcalar, toplam, secilen = [], 0, 0
    qc = {p: {"temizlenen": 0, "gecerli": 0} for p in POLLUTANTS}
    sutunlar = ["StationId", "ReadTime"] + POLLUTANTS
    for ch in pd.read_csv(SONDOSYA, chunksize=CHUNK, usecols=lambda c: c.strip() in sutunlar,
                          low_memory=False, encoding="utf-8-sig"):
        ch.columns = [c.strip() for c in ch.columns]
        toplam += len(ch)
        ch["ts"] = pd.to_datetime(ch["ReadTime"], errors="coerce")
        ch = ch[(ch["ts"] >= BASLANGIC) & (ch["ts"] < BITIS)].dropna(subset=["StationId"])
        if ch.empty:
            log(f"  okunan {toplam:,} satir (bu parcada donem disi)")
            continue
        ch = ch.drop_duplicates(subset=["StationId", "ts"], keep="first")
        secilen += len(ch)
        for p in POLLUTANTS:
            if p not in ch.columns:
                ch[p] = np.nan
                continue
            v = pd.to_numeric(ch[p], errors="coerce")
            gec = v.notna()
            sil = gec & (v < 0)
            for kod in BILINEN_KODLAR:
                sil |= gec & np.isclose(v, kod, atol=0.01)
            sil |= gec & (v > UST_ESIK[p])
            qc[p]["temizlenen"] += int(sil.sum())
            v = v.where(~sil)
            qc[p]["gecerli"] += int(v.notna().sum())
            ch[p] = v.astype("float32")
        ch["tarih"] = ch["ts"].dt.normalize()
        agg = {}
        for p in POLLUTANTS:
            agg[f"{p}__toplam"] = (p, "sum")
            agg[f"{p}__n"] = (p, "count")
        parcalar.append(ch.groupby(["StationId", "tarih"]).agg(**agg))
        log(f"  okunan {toplam:,} satir, donem ici {secilen:,}")

    log("  parcalar birlestiriliyor ...")
    g = pd.concat(parcalar).groupby(level=[0, 1]).sum(min_count=1)
    out = pd.DataFrame(index=g.index)
    for p in POLLUTANTS:
        n = g[f"{p}__n"].fillna(0)
        ort = g[f"{p}__toplam"] / n.replace(0, np.nan)
        out[p] = ort.where(n >= MIN_SAAT).astype("float32")
        out[f"{p}_saat"] = n.astype("int16")
    out = out.reset_index().rename(columns={"StationId": "istasyon_id"})
    out.to_parquet(A_CIKTI, index=False)
    with open(os.path.join(CIKTI, "A_qc_ozet.json"), "w", encoding="utf-8") as f:
        json.dump({"okunan_satir": toplam, "donem_ici_satir": secilen, "qc": qc}, f, indent=1)
    log(f"  A tamam: {len(out):,} istasyon-gun, {out['istasyon_id'].nunique()} istasyon")
    for p in POLLUTANTS:
        log(f"    {p:<5} gecerli gun={int(out[p].notna().sum()):,}  "
            f"istasyon(>=1 gecerli gun)={out.loc[out[p].notna(), 'istasyon_id'].nunique()}  "
            f"saatlik temizlenen={qc[p]['temizlenen']:,}")


# ----------------------------------------------------------------------------
# B) statik degiskenler
# ----------------------------------------------------------------------------
def zip_icinden_tif(zip_yolu, ad_parcasi=None):
    with zipfile.ZipFile(zip_yolu) as z:
        adlar = [n for n in z.namelist() if n.lower().endswith(".tif")]
    if ad_parcasi:
        adlar = [n for n in adlar if n.split("/")[-1].upper() == ad_parcasi.upper()]
    if not adlar:
        raise FileNotFoundError(f"{zip_yolu} icinde tif bulunamadi ({ad_parcasi})")
    return "/vsizip/" + zip_yolu.replace("\\", "/") + "/" + adlar[0]


def tamponlar_olustur(ist):
    import geopandas as gpd
    nok = gpd.GeoDataFrame(ist[["istasyon_id"]].copy(),
                           geometry=gpd.points_from_xy(ist["lon"], ist["lat"]), crs="EPSG:4326")
    nok_m = nok.to_crs(LAEA)
    tampon = {}
    for m in TAMPONLAR:
        t = nok_m.copy()
        t["geometry"] = nok_m.buffer(m, resolution=32)
        tampon[m] = t
    return nok, nok_m, tampon


def raster_nokta_degeri(yol, lon, lat):
    import rasterio
    with rasterio.open(yol) as r:
        vals = [v[0] for v in r.sample(list(zip(lon, lat)))]
        nod = r.nodata
    vals = np.array(vals, dtype="float64")
    if nod is not None:
        vals[np.isclose(vals, nod)] = np.nan
    return vals


def raster_tampon_toplam(yol, tampon_4326_list, istatistik="sum"):
    """Kucuk (yaklasik 1 km) rasterlarda tampon icindeki toplam/ortalama."""
    import rasterio
    from rasterio.mask import mask
    sonuc = []
    with rasterio.open(yol) as r:
        nod = r.nodata
        for geom in tampon_4326_list:
            try:
                arr, _ = mask(r, [geom], crop=True, all_touched=True, filled=False)
                a = arr[0].astype("float64")
                a = np.ma.masked_where(np.ma.getmaskarray(a) | (np.isclose(a, nod) if nod is not None else False), a)
                if a.count() == 0:
                    sonuc.append(np.nan)
                else:
                    sonuc.append(float(a.sum() if istatistik == "sum" else a.mean()))
            except Exception:
                sonuc.append(np.nan)
    return sonuc


def worldcover_oranlar(ist, tampon):
    import rasterio
    from rasterio.merge import merge
    from rasterio.features import geometry_mask
    log("  WorldCover sinif oranlari hesaplaniyor (10 m)...")
    tiles = {os.path.splitext(f)[0]: os.path.join(WORLDCOVER_KLASOR, f)
             for f in os.listdir(WORLDCOVER_KLASOR) if f.lower().endswith(".tif")}
    t5 = tampon[max(TAMPONLAR)].to_crs("EPSG:4326")
    kayit = []
    for i, row in ist.reset_index(drop=True).iterrows():
        g5 = t5.geometry.iloc[i]
        minx, miny, maxx, maxy = g5.bounds
        gerekli = set()
        for lat0 in range(int(np.floor(miny / 3) * 3), int(np.floor(maxy / 3) * 3) + 1, 3):
            for lon0 in range(int(np.floor(minx / 3) * 3), int(np.floor(maxx / 3) * 3) + 1, 3):
                ad = f"N{lat0:02d}E{lon0:03d}"
                if ad in tiles:
                    gerekli.add(tiles[ad])
        sat = {"istasyon_id": row["istasyon_id"]}
        if not gerekli:
            kayit.append(sat)
            continue
        srcs = [rasterio.open(p) for p in sorted(gerekli)]
        try:
            mozaik, tr = merge(srcs, bounds=(minx, miny, maxx, maxy))
        finally:
            for s in srcs:
                s.close()
        mozaik = mozaik[0]
        for m in TAMPONLAR:
            g = tampon[m].to_crs("EPSG:4326").geometry.iloc[i]
            icerde = ~geometry_mask([g], out_shape=mozaik.shape, transform=tr, invert=False)
            v = mozaik[icerde]
            v = v[v > 0]
            n = v.size
            for kod, ad in WC_SINIF.items():
                sat[f"wc_{ad}_{m}"] = float((v == kod).sum() / n) if n else np.nan
        kayit.append(sat)
        if (i + 1) % 50 == 0:
            log(f"    WorldCover {i + 1}/{len(ist)}")
    return pd.DataFrame(kayit)


def osm_degiskenleri(ist, nok_m, tampon):
    import geopandas as gpd
    from pyogrio import read_dataframe
    log("  OSM yollar okunuyor (zip icinden, bir kac dakika surebilir)...")
    zp = "/vsizip/" + OSM_ZIP.replace("\\", "/")
    yol = read_dataframe(zp + "/gis_osm_roads_free_1.shp", columns=["fclass"])
    yol = yol[~yol["fclass"].isin(YOL_HARIC)].to_crs(LAEA)
    yol["sinif"] = np.where(yol["fclass"].isin(YOL_ANA), "ana",
                            np.where(yol["fclass"].isin(YOL_ORTA), "orta", "diger"))
    log(f"    yol parcasi: {len(yol):,}")
    sidx = yol.sindex
    kayit = []
    ana = yol[yol["sinif"] == "ana"]
    ana_sidx = ana.sindex
    for i in range(len(ist)):
        sat = {"istasyon_id": ist["istasyon_id"].iloc[i]}
        p = nok_m.geometry.iloc[i]
        for m in TAMPONLAR:
            g = tampon[m].geometry.iloc[i]
            aday = yol.iloc[sidx.query(g, predicate="intersects")]
            kes = aday.geometry.intersection(g).length
            for s in ["ana", "orta", "diger"]:
                sat[f"yol_{s}_km_{m}"] = float(kes[aday["sinif"] == s].sum() / 1000.0)
            sat[f"yol_toplam_km_{m}"] = float(kes.sum() / 1000.0)
        j = ana_sidx.nearest(p, return_all=False)[1][0]
        sat["uzaklik_ana_yol_m"] = float(ana.geometry.iloc[j].distance(p))
        kayit.append(sat)
    df = pd.DataFrame(kayit)
    del yol, ana

    log("  OSM arazi kullanimi (sanayi, maden ocagi) okunuyor...")
    lu = read_dataframe(zp + "/gis_osm_landuse_a_free_1.shp", columns=["fclass"])
    lu = lu[lu["fclass"].isin(["industrial", "quarry"])].to_crs(LAEA)
    lu_sidx = lu.sindex
    ind = lu[lu["fclass"] == "industrial"]
    ind_sidx = ind.sindex if len(ind) else None
    ek = []
    for i in range(len(ist)):
        sat = {"istasyon_id": ist["istasyon_id"].iloc[i]}
        p = nok_m.geometry.iloc[i]
        for m in [1000, 3000, 5000]:
            g = tampon[m].geometry.iloc[i]
            aday = lu.iloc[lu_sidx.query(g, predicate="intersects")]
            alan = aday.geometry.intersection(g).area
            sat[f"sanayi_oran_{m}"] = float(alan[aday["fclass"] == "industrial"].sum() / g.area)
            sat[f"maden_ocagi_oran_{m}"] = float(alan[aday["fclass"] == "quarry"].sum() / g.area)
        if ind_sidx is not None:
            j = ind_sidx.nearest(p, return_all=False)[1][0]
            sat["uzaklik_sanayi_m"] = float(ind.geometry.iloc[j].distance(p))
        else:
            sat["uzaklik_sanayi_m"] = np.nan
        ek.append(sat)
    return df.merge(pd.DataFrame(ek), on="istasyon_id")


def asama_B():
    log("=" * 70)
    log("ASAMA B: istasyon statik degiskenleri")
    ist = istasyonlari_oku()
    nok, nok_m, tampon = tamponlar_olustur(ist)
    df = ist.copy()

    log("  GHS-POP ve GHS-BUILT (2020, 30 arcsec) tampon toplamlari...")
    pop = zip_icinden_tif(GHS_POP_ZIP)
    blt = zip_icinden_tif(GHS_BUILT_ZIP)
    for m in [1000, 3000, 5000]:
        g4326 = list(tampon[m].to_crs("EPSG:4326").geometry)
        df[f"nufus_{m}"] = raster_tampon_toplam(pop, g4326, "sum")
        alan_m2 = tampon[m].geometry.area.values
        df[f"yapilasma_oran_{m}"] = np.array(raster_tampon_toplam(blt, g4326, "sum")) / alan_m2
    df["nufus_piksel"] = raster_nokta_degeri(pop, ist["lon"], ist["lat"])

    log("  Gunes atlasi (GHI, TEMP) ve ruzgar atlasi nokta degerleri...")
    df["gsa_ghi_kwh_m2_gun"] = raster_nokta_degeri(zip_icinden_tif(GUNES_ZIP, "GHI.tif"), ist["lon"], ist["lat"])
    df["gsa_sicaklik_uzun_donem_c"] = raster_nokta_degeri(zip_icinden_tif(GUNES_ZIP, "TEMP.tif"), ist["lon"], ist["lat"])
    df["gwa_ruzgar_100m_ms"] = raster_nokta_degeri(RUZGAR_TIF, ist["lon"], ist["lat"])

    df = df.merge(worldcover_oranlar(ist, tampon), on="istasyon_id", how="left")
    df = df.merge(osm_degiskenleri(ist, nok_m, tampon), on="istasyon_id", how="left")

    topo = pd.read_csv(TOPO).drop(columns=["istasyon", "il"], errors="ignore")
    sant = pd.read_csv(SANTRAL).drop(columns=["istasyon", "il"], errors="ignore")
    sant["uzaklik_komur_santral_m"] = sant["uzaklik_komur_santral_m"].fillna(300000.0)
    sant["uzaklik_fosil_santral_m"] = sant["uzaklik_fosil_santral_m"].fillna(300000.0)
    df = df.merge(topo, on="istasyon_id", how="left").merge(sant, on="istasyon_id", how="left")

    df.to_parquet(B_CIKTI, index=False)
    log(f"  B tamam: {df.shape[0]} istasyon x {df.shape[1]} sutun")


# ----------------------------------------------------------------------------
# C) yillik degiskenler
# ----------------------------------------------------------------------------
def tuik_oku():
    t = pd.read_csv(TUIK, sep="\t", encoding="cp1254")
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
    t = t.rename(columns=yeni)
    t = t.rename(columns={"YIL": "yil", "BÖLGE ADI": "il_tuik"})
    tutulan = ["yil", "il_tuik"] + sorted(set(yeni.values()))
    t = t[tutulan].copy()
    for c in tutulan[2:]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    for c in ["tuik_otoyol_km", "tuik_ucak_inis_kalkis"]:
        t[c] = t[c].fillna(0.0)
    t["il_anahtar"] = t["il_tuik"].map(il_norm)
    # 2026 satiri ekle, sonra il bazinda son bilinen degerle ileri doldur
    son = t[t["yil"] == t["yil"].max()].copy()
    son["yil"] = 2026
    t = pd.concat([t, son], ignore_index=True).sort_values(["il_anahtar", "yil"])
    deg = [c for c in t.columns if c.startswith("tuik_")]
    t[deg] = t.groupby("il_anahtar")[deg].ffill()
    # turetilmis degiskenler
    t["tuik_kisi_basi_mesken_elektrik_kwh"] = t["tuik_mesken_elektrik_mwh"] * 1000 / t["tuik_nufus"]
    t["tuik_bin_kisi_agir_tasit"] = (t[["tuik_kamyon", "tuik_kamyonet", "tuik_otobus", "tuik_minibus"]]
                                     .sum(axis=1, min_count=1) * 1000 / t["tuik_nufus"])
    t["tuik_bin_kisi_imalat_girisim"] = t["tuik_girisim_imalat"] * 1000 / t["tuik_nufus"]
    return t.drop(columns=["il_tuik"])


def asama_C():
    log("=" * 70)
    log("ASAMA C: yillik degiskenler (NDVI, gece isigi, TUIK)")
    ist = istasyonlari_oku()
    nd = pd.read_csv(NDVI)
    nd = nd.pivot_table(index=["istasyon_id", "yil"], columns="tampon_m",
                        values=["ndvi", "gece_isigi"])
    nd.columns = [f"{a}_{int(b)}" for a, b in nd.columns]
    nd = nd.reset_index()
    son = nd[nd["yil"] == nd["yil"].max()].copy()
    son["yil"] = 2026
    nd = pd.concat([nd, son], ignore_index=True)

    tu = tuik_oku()
    yillar = pd.DataFrame({"yil": range(BASLANGIC.year, 2027)})
    taban = ist[["istasyon_id", "il_anahtar"]].merge(yillar, how="cross")
    df = taban.merge(nd, on=["istasyon_id", "yil"], how="left")
    df = df.merge(tu, on=["il_anahtar", "yil"], how="left")
    eslesmeyen = df.loc[df["tuik_nufus"].isna(), "il_anahtar"].unique()
    if len(eslesmeyen):
        log(f"  UYARI: TUIK ile eslesmeyen il/yil: {list(eslesmeyen)[:10]}")
    df.drop(columns=["il_anahtar"]).to_parquet(C_CIKTI, index=False)
    log(f"  C tamam: {df.shape}")


# ----------------------------------------------------------------------------
# D) gunluk panel
# ----------------------------------------------------------------------------
def asama_D():
    log("=" * 70)
    log("ASAMA D: gunluk panel birlestiriliyor")
    a = pd.read_parquet(A_CIKTI)
    a["tarih"] = pd.to_datetime(a["tarih"])

    s5 = pd.read_csv(S5P, usecols=["istasyon_id", "tarih", "tropospheric_NO2_column_number_density",
                                   "SO2_column_number_density"])
    s5 = s5.rename(columns={"tropospheric_NO2_column_number_density": "s5p_no2_mol_m2",
                            "SO2_column_number_density": "s5p_so2_mol_m2"})
    ao = pd.read_csv(AOD, usecols=["istasyon_id", "tarih", "mean"]).rename(columns={"mean": "modis_aod550"})
    ao["modis_aod550"] = ao["modis_aod550"] * 0.001
    er = pd.read_csv(ERA5)
    er = er.drop(columns=["istasyon", "il"], errors="ignore")
    t = er["temperature_2m"] - 273.15
    td = er["dewpoint_temperature_2m"] - 273.15
    era = pd.DataFrame({
        "istasyon_id": er["istasyon_id"], "tarih": er["tarih"],
        "era5_sicaklik_c": t,
        "era5_cig_noktasi_c": td,
        "era5_bagil_nem_pct": 100 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t)),
        "era5_ruzgar_hiz_ms": np.hypot(er["u_component_of_wind_10m"], er["v_component_of_wind_10m"]),
        "era5_ruzgar_u_ms": er["u_component_of_wind_10m"],
        "era5_ruzgar_v_ms": er["v_component_of_wind_10m"],
        "era5_basinc_hpa": er["surface_pressure"] / 100.0,
        "era5_yagis_mm": er["total_precipitation_sum"] * 1000.0,
        "era5_gunes_mj_m2": er["surface_solar_radiation_downwards_sum"] / 1e6,
    })
    for d in (s5, ao, era):
        d["tarih"] = pd.to_datetime(d["tarih"])

    ist = istasyonlari_oku()
    # tam iskelet: 356 istasyon x tum gunler
    gunler = pd.DataFrame({"tarih": pd.date_range(BASLANGIC, BITIS - pd.Timedelta(days=1), freq="D")})
    p = ist[["istasyon_id"]].merge(gunler, how="cross")
    p = p.merge(a, on=["istasyon_id", "tarih"], how="left")
    p = p.merge(s5, on=["istasyon_id", "tarih"], how="left")
    p = p.merge(ao, on=["istasyon_id", "tarih"], how="left")
    p = p.merge(era, on=["istasyon_id", "tarih"], how="left")
    p["yil"] = p["tarih"].dt.year
    p["ay"] = p["tarih"].dt.month
    p["yilin_gunu"] = p["tarih"].dt.dayofyear
    p["haftanin_gunu"] = p["tarih"].dt.dayofweek
    p = p.merge(pd.read_parquet(C_CIKTI), on=["istasyon_id", "yil"], how="left")
    p = p.merge(pd.read_parquet(B_CIKTI), on="istasyon_id", how="left")

    # kirletici olcumu hic olmayan istasyon-gunleri cikar (model icin gereksiz)
    p = p[p[POLLUTANTS].notna().any(axis=1)].reset_index(drop=True)
    p.to_parquet(D_CIKTI, index=False)
    log(f"  D tamam: {p.shape[0]:,} satir x {p.shape[1]} sutun -> {D_CIKTI}")


# ----------------------------------------------------------------------------
# E) kontrol raporu
# ----------------------------------------------------------------------------
def asama_E():
    log("=" * 70)
    log("ASAMA E: kontrol raporu ve veri sozlugu")
    p = pd.read_parquet(D_CIKTI)
    b = pd.read_parquet(B_CIKTI)
    satir = []
    for c in p.columns:
        s = p[c]
        bilgi = {"sutun": c, "tip": str(s.dtype), "dolu_yuzde": round(100 * s.notna().mean(), 2)}
        if pd.api.types.is_numeric_dtype(s):
            bilgi.update({"ortalama": s.mean(), "min": s.min(), "medyan": s.median(), "maks": s.max()})
        satir.append(bilgi)
    pd.DataFrame(satir).to_csv(SOZLUK, index=False, encoding="utf-8-sig")

    r = []
    r.append("CBS MAKALESI - HAZIRLIK RAPORU")
    r.append("=" * 60)
    r.append(f"Panel: {len(p):,} istasyon-gun, {p['istasyon_id'].nunique()} istasyon, "
             f"{p['tarih'].min().date()} .. {p['tarih'].max().date()}, {p.shape[1]} sutun")
    r.append("")
    r.append("Kirletici bazinda (gunluk ortalama, >=18 saat):")
    for k in POLLUTANTS:
        gec = p[k].notna()
        ist_gun = p.loc[gec].groupby("istasyon_id").size()
        r.append(f"  {k:<5} gecerli gun={int(gec.sum()):>9,}  istasyon={ist_gun.size:>3}  "
                 f">=365 gun olan istasyon={int((ist_gun >= 365).sum()):>3}  "
                 f"ortalama={p[k].mean():.2f}  medyan={p[k].median():.2f}")
    r.append("")
    r.append("Gunluk ortak degiskenlerin dolulugu (%):")
    for c in ["s5p_no2_mol_m2", "s5p_so2_mol_m2", "modis_aod550", "era5_sicaklik_c",
              "era5_bagil_nem_pct", "ndvi_1000", "gece_isigi_1000", "tuik_nufus"]:
        if c in p:
            r.append(f"  {c:<28} {100 * p[c].notna().mean():6.2f}")
    r.append("")
    r.append("Statik degiskenlerde eksik deger olan sutunlar (356 istasyon uzerinden):")
    eksik = b.isna().sum()
    eksik = eksik[eksik > 0]
    r.append("  yok" if eksik.empty else "\n".join(f"  {k}: {v}" for k, v in eksik.items()))
    r.append("")
    r.append("Makullik kontrolleri:")
    r.append(f"  ERA5 sicaklik araligi: {p['era5_sicaklik_c'].min():.1f} .. {p['era5_sicaklik_c'].max():.1f} C")
    r.append(f"  ERA5 bagil nem araligi: {p['era5_bagil_nem_pct'].min():.1f} .. {p['era5_bagil_nem_pct'].max():.1f} %")
    r.append(f"  MODIS AOD araligi: {p['modis_aod550'].min():.3f} .. {p['modis_aod550'].max():.3f}")
    wc = [c for c in b.columns if c.startswith("wc_") and c.endswith("_1000")]
    if wc:
        top = b[wc].sum(axis=1, min_count=1)
        r.append(f"  WorldCover 1 km sinif oranlari toplami: min={top.min():.3f} maks={top.max():.3f} (1'e yakin olmali)")
    if "yol_toplam_km_1000" in b:
        r.append(f"  1 km tampon yol uzunlugu (km): medyan={b['yol_toplam_km_1000'].median():.1f} "
                 f"maks={b['yol_toplam_km_1000'].max():.1f}")
    if "nufus_1000" in b:
        r.append(f"  1 km tampon nufus: medyan={b['nufus_1000'].median():,.0f} maks={b['nufus_1000'].max():,.0f}")
    with open(RAPOR, "w", encoding="utf-8") as f:
        f.write("\n".join(r) + "\n")
    log("\n".join(r))
    log("E tamam.")


# ----------------------------------------------------------------------------
ASAMALAR = [("A", asama_A, A_CIKTI), ("B", asama_B, B_CIKTI), ("C", asama_C, C_CIKTI),
            ("D", asama_D, D_CIKTI), ("E", asama_E, RAPOR)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yeniden", default="", help="Yeniden uretilecek asama harfi (A-E)")
    args = ap.parse_args()
    if args.yeniden:
        harfler = [h for h, _, _ in ASAMALAR]
        bas = harfler.index(args.yeniden.upper())
        for h, _, yol in ASAMALAR[bas:]:
            if os.path.exists(yol):
                os.remove(yol)
                log(f"{h} ciktisi silindi (yeniden uretilecek)")
    t0 = time.time()
    for harf, fonk, yol in ASAMALAR:
        if os.path.exists(yol):
            log(f"ASAMA {harf}: cikti zaten var, atlaniyor ({os.path.basename(yol)})")
            continue
        durum_yaz(harf, "calisiyor")
        try:
            fonk()
            durum_yaz(harf, "tamam")
        except Exception as e:
            import traceback
            durum_yaz(harf, f"HATA: {e}")
            log(f"ASAMA {harf} HATA: {e}\n{traceback.format_exc()}")
            sys.exit(1)
    durum_yaz("TUMU", f"tamam ({(time.time() - t0) / 60:.1f} dk)")
    log(f"TUM ASAMALAR TAMAM ({(time.time() - t0) / 60:.1f} dk)")


if __name__ == "__main__":
    main()
