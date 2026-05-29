# DPM-Solver++(2M) vs Euler — A/B Timing Report

**Date:** 2026-05-29 18:34  
**Model:** Lance-3B-bf16  
**Seed:** 42  
**Resolution:** 768×768  
**CFG scale:** 4.0  
**Pipeline load time:** 20.2s  

## Grid

![A/B grid](ab_grid.png)

## Per-image timings

| Prompt ID | Scheduler | Steps | Time (s) | vs Euler |
|-----------|-----------|-------|----------|----------|
| p01_fox_grass | DPM++ 12-step | 12 | 80.2 | **2.77×** faster |
| p01_fox_grass | Euler 30-step | 30 | 221.8 | baseline |
| p02_chef | DPM++ 12-step | 12 | 148.3 | **1.61×** faster |
| p02_chef | Euler 30-step | 30 | 239.5 | baseline |
| p03_coastline | DPM++ 12-step | 12 | 102.8 | **2.20×** faster |
| p03_coastline | Euler 30-step | 30 | 226.6 | baseline |
| p04_anime_street | DPM++ 12-step | 12 | 149.2 | **1.68×** faster |
| p04_anime_street | Euler 30-step | 30 | 251.3 | baseline |

## Summary

| Metric | DPM++ 12-step | Euler 30-step |
|--------|--------------|---------------|
| Avg time/image | 120.1s | 234.8s |
| Speedup | **1.95×** | 1.00× |
| Total (4 images) | 480s | 939s |

## Individual images

- **p01_fox_grass / DPM++ 12-step** — `p01_fox_grass_dpm12.png` (80.2s)
- **p01_fox_grass / Euler 30-step** — `p01_fox_grass_euler30.png` (221.8s)
- **p02_chef / DPM++ 12-step** — `p02_chef_dpm12.png` (148.3s)
- **p02_chef / Euler 30-step** — `p02_chef_euler30.png` (239.5s)
- **p03_coastline / DPM++ 12-step** — `p03_coastline_dpm12.png` (102.8s)
- **p03_coastline / Euler 30-step** — `p03_coastline_euler30.png` (226.6s)
- **p04_anime_street / DPM++ 12-step** — `p04_anime_street_dpm12.png` (149.2s)
- **p04_anime_street / Euler 30-step** — `p04_anime_street_euler30.png` (251.3s)
