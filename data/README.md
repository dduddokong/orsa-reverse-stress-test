# Data

The model's input data lives in [`../model/data/`](../model/data/), not here: the `model` package loads it by path and the tests import that package, so the files stay next to the code.

| File | Contents | Source |
|---|---|---|
| `disclosures_2026h1.csv` | K-ICS capital, required capital, sub-risks, diversification effect, sensitivities per insurer (base date 2026-06-30, KRW 100m) | Insurers' periodic management disclosures (정기경영공시) and IR materials |
| `kics_walk.csv` | Quarter-on-quarter K-ICS ratio walk by factor (pp) | Insurers' IR materials |
| `market_levels.csv` | KTB / corporate bond yields, KOSPI levels | Bank of Korea ECOS |
| `va_history.csv` | Volatility adjustment and liquidity premium history | FSS volatility-adjustment publications, BoK ECOS |
| `liability_fit.json` | Fitted model-point parameters (output of `python -m model.fit_liability`) | Generated |

Every row records its source URL and page. Some figures also use FSS FISIS and a few news items. No proprietary or non-public data is used. Raw downloaded PDFs/spreadsheets are not committed (`data/raw/` is git-ignored).
