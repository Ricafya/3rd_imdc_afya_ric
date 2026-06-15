# 3rd IMDC 2026 — Afya  Dengue Forecasting Model

Submission of the **Afya** team for the **3rd InfoDengue–Mosqlimate Dengue Challenge (IMDC 2026)**.

> **Challenges entered:** Mandatory — Dengue (state / UF level).


## 1. Team and Contributors

- **Team name:** Afya
- **Team leader:** Dayanna Quintanilha — Universidade Federal Fluminense (UFF), Niterói, RJ, Brazil & Research and Innovation Center, Afya, São Paulo, SP, Brazil [dayanna.quintanilha@afya.com.br]
- **Contributors:**
  - Dayanna Quintanilha — Universidade Federal Fluminense (UFF), Niterói, RJ, Brazil & Research and Innovation Center, Afya, São Paulo, SP, Brazil
  - Marcela Motta — Research and Innovation Center, Afya, São Paulo, SP, Brazil
  - Danielly Xavier — Research and Innovation Center, Afya, São Paulo, SP, Brazil
  - Eduardo Moura — Research and Innovation Center, Afya, São Paulo, SP, Brazil
  - Angélica Caseri — Institute of Mathematical and Computer Sciences (ICMC), Universidade de São Paulo (USP), São Paulo, SP, Brazil
  - Julia Valentim — Research and Innovation Center, Afya, São Paulo, SP, Brazil

## 2. Repository Structure

| Path | Description |
| --- | --- |
| `Demo Notebooks/` | Reference notebooks (Python / R) from the IMDC template for submitting forecasts via `mosqlient`. |
| `data/` | [Raw and/or processed datasets — or note that data is pulled from the Mosqlimate API / FTP at runtime.] |
| `src/` | [Model code: preprocessing, training, forecasting.] |
| `notebooks/` | [Exploratory analysis / model development.] |
| `pyproject.toml` | Project dependencies. |
| `README.md` | This file. |

_Adjust the table to match your actual layout._

## 3. Libraries and Dependencies

- Python 3.10+
- `mosqlient` — data access and forecast submission to the Mosqlimate platform
- 

Installation:



## 4. Data and Variables

- **Datasets used:** [e.g. InfoDengue probable cases (use the `casprov` column), climate and demographic data from the Mosqlimate FTP / API.]
- **Variables:** [dengue cases, temperature, precipitation, relative humidity, ...]
- **Pre-processing:** [Cleaning, aggregation to weekly resolution, missing-value handling, etc.]
- **Variable selection:** [How features were chosen — point to the relevant code, e.g. `src/features.py`.]

## 5. Model Training

- **Methodology:** [statistical / mechanistic-epidemiological / machine learning — describe the approach.]
- **Training procedure:** [How the model was fit; hyperparameter optimization, if any.]
- **How to reproduce:** [Commands / scripts to run training and generate forecasts, e.g. `python src/train.py` then `python src/forecast.py`.]
- **Temporal resolution:** Weekly (required by the challenge).

## 6. Data Usage Restriction


## 7. Predictive Uncertainty


The challenge requires, for each target, the **median (0.5 quantile)** plus the **50%, 80%, 90% and 95% predictive intervals**, i.e. the following quantiles:

| Interval | Lower quantile | Upper quantile |
| --- | --- | --- |
| Median | — | 0.500 |
| 50% | 0.250 | 0.750 |
| 80% | 0.100 | 0.900 |
| 90% | 0.050 | 0.950 |
| 95% | 0.025 | 0.975 |

## 8. References

Palmer DQ, Motta M, Moura E, Xavier D, Schittine G, Caseri A, et al. Dengue hospitalizations in Brazil: forecasting with climatic and physicians' digital search data under real-world reporting delays. PLOS Digit Health. 2026;5(5):e0001206. https://doi.org/10.1371/journal.pdig.0001206

