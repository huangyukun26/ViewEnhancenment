# Reference-consistency short cycle

All commands run from the repository root. Previous `restoration_v2` results
are kept in their original directories; this round writes under
`research/outputs/restoration_v2/reference_consistency/`.

```powershell
& 'G:\fishnet\.venv_cuda\Scripts\python.exe' research/restoration_v2/reference_consistency.py
```

RoMa v2 official implementation, fast setting (512px, batch 1):

```powershell
& 'C:\Users\LENOVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' research/restoration_v2/roma_reference.py --target <target.png> --reference <reference.png> --out .research_cache/roma_smoke.json
& 'C:\Users\LENOVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' research/restoration_v2/roma_showcase.py --limit 8 --confidence 0.5
```

MatrixCity small-city `road_down` partial-range experiment (the RGB range,
pose JSON and generated images stay in `.research_cache/` and are not part of
the repository):

```powershell
& 'G:\fishnet\.venv_cuda\Scripts\python.exe' research/restoration_v2/matrixcity_small_eval.py --archive .research_cache/matrixcity/down_dense_512mb.bin --max-images 300 --pairs 100
```

Sources: [RoMa v2](https://github.com/Parskatt/RoMaV2) and
[MatrixCity](https://github.com/city-super/MatrixCity). MatrixCity data is
used only for the held-out GT sanity experiment; Open3DHK deployment inputs
remain RGB, mask and a nearby reference image.
