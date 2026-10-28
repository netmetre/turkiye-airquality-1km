# turkiye-airquality-1km

Code, 1 km maps and result tables for the study

> Gültekin, F. *Spatially validated machine learning for nationwide multi-pollutant air quality mapping and population exposure assessment in Türkiye*. Submitted to Environmental Modelling & Software.

The workflow combines 354 stations of the National Air Quality Monitoring Network of Türkiye (2019-2026) with daily Sentinel-5P, MAIAC and ERA5-Land predictors and with static geographic predictors (SRTM, ESA WorldCover, GHSL, VIIRS, MODIS NDVI, OpenStreetMap, power plants). It compares eight interpolation and machine learning methods under random, station-blocked and province-blocked cross-validation. It then produces 1 km maps of PM10, NO2, SO2 and O3 for 2025 and computes population exposure and monitoring gaps.

## Contents

| Folder | Content |
|---|---|
| `gee/` | Google Earth Engine scripts (Code Editor) |
| `code/` | Python scripts, run in the order below |
| `maps/2025/` | 1 km GeoTIFFs in µg m-3 for each pollutant: `*_annual` (annual mean), `*_ensemble_sd` (standard deviation of five bootstrap ensemble members), `*_DJF` and `*_JJA` (winter and summer means) |
| `tables/` | Cross-validation metrics, exposure and monitoring-gap tables (CSV) |

Map projection: Lambert azimuthal equal-area, `+proj=laea +lat_0=39 +lon_0=35 +x_0=0 +y_0=0 +datum=WGS84 +units=m`, 1000 m cells.

### Tables

| File | Content |
|---|---|
| `cv_metrics_summary.csv` | Out-of-fold metrics per pollutant, design and method (daily and station-mean) |
| `cv_metrics_by_fold.csv` | The same per fold |
| `exposure_national_2025.csv` | Area and population-weighted means, share of population above WHO 2021 guideline values and EU 2030 limits, with ±1 ensemble SD |
| `exposure_province_2025.csv` | Population-weighted means and population above the WHO guideline by province |
| `exposure_district_2025.csv` | The same by district (OCHA COD-AB p-codes), with station counts |
| `monitoring_gaps_2025.csv` | Districts and population without a station, population beyond 10, 25 and 50 km |
| `station_siting_priority_2025.csv` | Ranking of districts without a station |

## Workflow

1. `gee/gee_istasyon_uydu_verisi.js`, `gee/gee_istasyon_statik_degiskenler.js`, `gee/gee_duzeltme_aod_era5.js`: satellite, reanalysis and static predictors at the stations.
2. `code/cbs_hazirlik.py`: quality control of hourly data, daily means (at least 18 valid hours), station-day panel.
3. `code/cbs_kesifsel_analiz.py`: Moran's I, Getis-Ord Gi*, semivariograms.
4. `code/cbs_modelleme.py`: model comparison (IDW, ridge LUR, random forest, XGBoost, LightGBM, LightGBM+IDW, GNN; ordinary kriging of station means) under three cross-validation designs, and SHAP. A GPU is used for XGBoost and the GNN when available.
5. `code/cbs_sonuc_sekilleri.py`: comparison figures.
6. `gee/gee_izgara_degiskenleri.js`, `gee/gee_izgara_ghs_duzeltme.js`: the same predictors on a 30 arcsec grid, re-derived at the stations.
7. `code/cbs_izgara.py`: 1 km grid, OpenStreetMap rasters, mapping model, re-validation, bootstrap ensemble, bias correction, GeoTIFF output.
8. `code/cbs_nufus_duzeltme.py`: rescales GHS-POP 2020 cell population to 2022 census district totals.
9. `code/cbs_harita_sekilleri.py`, `code/cbs_maruziyet.py`, `code/cbs_ek_analizler.py`, `code/cbs_makale_ek_sekiller.py`: maps, exposure, distance diagnostics, sensitivity analyses and remaining figures.

The Python scripts expect the repository root to contain `veri/` (input data) and write to `islenmis/` (outputs). Each script writes a log and skips stages whose outputs already exist. The map year and day sampling are set at the top of `gee/gee_izgara_degiskenleri.js` (`HARITA_YILI`, `ADIM_GUN`) and `code/cbs_izgara.py`, so the maps can be rebuilt for another year. In the Earth Engine scripts, replace `<your-earth-engine-project>` with your own project. Variable and comment names in the code are partly in Turkish; Table S1 of the paper describes each predictor.

## Requirements

Python 3.11 with the packages in `requirements.txt`, and a Google Earth Engine account. Tested on Windows 11 with an AMD Ryzen 9 8940HX, 32 GB RAM and an NVIDIA RTX 5070 Laptop GPU. The model comparison took about 4 h and the mapping about 75 min.

## Input data

- Hourly air quality data: Ministry of Environment, Urbanisation and Climate Change (https://sim.csb.gov.tr). These data are not redistributed here.
- District boundaries: OCHA COD-AB Türkiye (https://data.humdata.org/dataset/cod-ab-tur).
- District population for 2022: OCHA/UNFPA COD-PS Türkiye, from TurkStat (https://data.humdata.org/dataset/cod-ps-tur).
- All other inputs are open; see Table 2 of the paper. OpenStreetMap data © OpenStreetMap contributors, ODbL.

## License

Code: MIT License (see `LICENSE`). Maps and tables in `maps/` and `tables/`: CC BY 4.0.

## Citation

See `CITATION.cff`. Please cite the archived release (Zenodo DOI) and the paper.

## Contact

Fatih Gültekin, Karabük University, fatihgultekin@karabuk.edu.tr, ORCID 0000-0003-1112-9830
