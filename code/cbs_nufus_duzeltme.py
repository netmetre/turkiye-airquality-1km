"""Hucre nufusunun TUIK ilce nufusuna gore olceklenmesi (dasimetrik duzeltme).

GHS-POP 2020 bazi ilcelerde nufusu sanayi yapilarina da dagitiyor (or. Yumurtalik).
Her hucrenin GHS nufusu, ilcesinin TUIK toplamina esitlenecek sekilde olceklenir:
    nufus_duz = nufus_ghs * TUIK_ilce / sum(nufus_ghs in ilce)
GHS toplami sifir olan ilcelerde TUIK nufusu ilcenin hucrelerine esit dagitilir.

Girdi : islenmis/izgara/H_harita_2025.parquet
        veri/TUR_ADM2_OCHA_CODAB.geojson   (OCHA COD-AB, shapeID = adm2 pcode)
        veri/tur_admpop_adm2_2022.csv      (OCHA COD-PS, TUIK 2022)
Cikti : islenmis/izgara/N_nufus_2025.parquet (hucre_id, ilce_id, nufus_ghs, nufus_duz)
Kullanim: python cbs_nufus_duzeltme.py [kok_klasor]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

KOK = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
LAEA = "+proj=laea +lat_0=39 +lon_0=35 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
ILCE = KOK / "veri" / "TUR_ADM2_OCHA_CODAB.geojson"
NUF = KOK / "veri" / "tur_admpop_adm2_2022.csv"
CIKTI = KOK / "islenmis" / "izgara" / "N_nufus_2025.parquet"

H = pd.read_parquet(KOK / "islenmis" / "izgara" / "H_harita_2025.parquet", columns=["hucre_id", "lon", "lat", "nufus_hucre"])
H["nufus_ghs"] = H["nufus_hucre"].fillna(0).clip(lower=0)
ilce = gpd.read_file(ILCE).to_crs(LAEA)[["shapeName", "shapeID", "geometry"]]
hucre = gpd.GeoDataFrame(geometry=gpd.points_from_xy(H.lon, H.lat), crs=4326).to_crs(LAEA)
es = gpd.sjoin(hucre, ilce, how="left", predicate="within")
es = es[~es.index.duplicated()]
H["ilce_id"] = es["shapeID"].values
bos = H.ilce_id.isna().values
if bos.any():
    j = gpd.sjoin_nearest(hucre[bos], ilce, how="left")
    H.loc[bos, "ilce_id"] = j[~j.index.duplicated()]["shapeID"].values

t = pd.read_csv(NUF)
pc = [c for c in t.columns if c.upper() in ("ADM2_PCODE", "ADM2PCODE", "PCODE")][0]
tl = [c for c in t.columns if c.upper() in ("T_TL", "TOTAL", "POPULATION", "POP", "T")]
tl = tl[0] if tl else t.select_dtypes("number").sum().idxmax()
tuik = t.set_index(pc)[tl].astype(float)

ghs = H.groupby("ilce_id")["nufus_ghs"].sum()
say = H.groupby("ilce_id").size()
oran = (tuik / ghs).replace([np.inf, -np.inf], np.nan)
H["nufus_duz"] = H["nufus_ghs"] * H["ilce_id"].map(oran)
sifir = H["ilce_id"].map(ghs).eq(0)
H.loc[sifir, "nufus_duz"] = (H.loc[sifir, "ilce_id"].map(tuik) / H.loc[sifir, "ilce_id"].map(say)).values
eksik = H["nufus_duz"].isna()
H.loc[eksik, "nufus_duz"] = H.loc[eksik, "nufus_ghs"]  # TUIK kaydi olmayan ilce: GHS korunur

H[["hucre_id", "ilce_id", "nufus_ghs", "nufus_duz"]].to_parquet(CIKTI, index=False)
k = pd.DataFrame({"ad": ilce.set_index("shapeID")["shapeName"], "ghs": ghs, "tuik": tuik, "oran": oran})
k.sort_values("oran").to_csv(CIKTI.with_name("N_nufus_ilce_karsilastirma.csv"), encoding="utf-8-sig")
print(f"TUIK kolonu: {pc}/{tl}; eslesmeyen ilce: {int(k.tuik.isna().sum())}, TUIK'te olup izgarada olmayan: {int(k.ghs.isna().sum())}")
print(f"GHS toplam {H.nufus_ghs.sum() / 1e6:.2f} M, duzeltilmis toplam {H.nufus_duz.sum() / 1e6:.2f} M")
print("oran dagilimi:", k.oran.describe().round(2).to_dict())
print(k.sort_values("oran").head(8).to_string())
print(k.sort_values("oran").tail(5).to_string())
