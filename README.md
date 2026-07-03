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
  - Ronaldo Gismondi - Universidade Federal Fluminense (UFF), Niterói, RJ, Brazil & Research and Innovation Center, Afya, São Paulo, SP, Brazil

## 2. Repository Structure

| Path | Description |
| --- | --- |
| `data_raw/` | [Raw datasets used in the model.] |
| `notebooks/` | [Notebooks used in the training part.] |
| `src/` | [Model code: preprocessing, training, forecasting.] |
| `scr/1_preprocessing` | [Preprocessing of the raw datas to be ready to training the models.] |
| `scr/2_model` | [Training of the model.] |
| `scr/2_model/model` | [Models of each uf.] |
| `pyproject.toml` | Project dependencies. |
| `README.md` | This file. |

_Adjust the table to match your actual layout._

## 3. Libraries and Dependencies

- Python 3.10+
- numpy==1.24.4
- pandas==1.4.2
- torch==2.4.1
- scikit-learn==1.3.2
- matplotlib==3.7.5


## 4. Data and Variables

- **Datasets used:** [Dengue cases (use the `casos` column), climate, forecasting climate and demographic data and access to dengue subject in Afya plataform obtained from the Mosqlimate FTP / API.]
- **Variables:** [dengue cases, temperature mean, precipitation mean, pressure mean, relative humidity mean, thermal range, rainy days, temperature mean 2 months forecasting, humidity mean 2 months forecasting, precipitation mean 2 months forecasting, Afya access count.]
- **Pre-processing:** [Aggregation to weekly resolution, exclude Espírito Santo from data,
calculate the incidence normalized: casos_norm = casos / population × 100.000, merge from datas in a unique dataset.]
- **Variable selection:** To avoid using collinear variables, we chose one climate variable from each category. In this case, we selected the mean.

## 5. Model Training

- **Methodology:** LSTM model forecasting.
- **Training procedure:** Training procedure
Optimizer: Adam (lr=1e-3)
Loss: MSE
Gradient clipping: max_norm=1.0
Early stopping: patience of 15 epochs on validation loss
Validation set: last 10% of training sequences (not shuffled)
Best model weights restored after early stopping
No hyperparameter optimization was performed — all values were set manually and held fixed across all UFs and splits.
- **How to reproduce:** [The command to run training and generate forecasts is `python src/2_model/training.py`.]
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

