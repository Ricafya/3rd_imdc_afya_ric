"""
LSTM com variaveis exogenas -- Dengue x Clima x AFYA (intervalos de previsao por percentil).

Versao em script de training.ipynb. Para cada combinacao split x UF:
  - treina um modelo LSTM independente e salva os pesos em models/
  - gera a previsao recursiva (mediana + intervalos de percentil via MC Dropout)

Ao final, escreve:
  - predictions.csv                          -> tabela completa de previsoes (mesmo layout ja usado)
  - lstm_climate_afya_percentil_submit.csv   -> tabela de submissao (sem casos_real)
  - lstm_climate_afya_percentil_metrics.csv  -> MAE/RMSE/coverage por split x UF
  - lstm_climate_afya_percentil_plots.pdf    -> graficos por UF
  - models/lstm_split{split}_{uf}.pt         -> um arquivo de modelo treinado por split x UF
"""
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x

warnings.filterwarnings('ignore')

# --- Features ---
TARGET          = 'casos'
EXOG_HISTORICAL = ['temp_med', 'precip_med', 'pressure_med', 'rel_humid_med', 'thermal_range', 'rainy_days']
EXOG_FORECAST   = ['temp_med_2m', 'umid_med_2m', 'precip_tot_2m']
EXOG_AFYA       = ['afya_access_count']
EXOG_FEATURES   = EXOG_HISTORICAL + EXOG_FORECAST + EXOG_AFYA
ALL_FEATURES    = [TARGET] + EXOG_FEATURES  # indice 0 = casos

# --- Hiperparametros ---
LOOKBACK    = 52
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2
EPOCHS      = 100
LR          = 1e-3
BATCH_SIZE  = 32
PATIENCE    = 15
MC_SAMPLES  = 200   # amostras MC Dropout para estimar a distribuicao de previsoes

# Percentis gerados para cada intervalo de previsao
PERCENTILE_MAP = {
    'lower_95': 2.5,
    'lower_90': 5.0,
    'lower_80': 10.0,
    'lower_50': 25.0,
    'pred':     50.0,
    'upper_50': 75.0,
    'upper_80': 90.0,
    'upper_90': 95.0,
    'upper_95': 97.5,
}
PI_COLS = list(PERCENTILE_MAP.keys())

SPLITS = [1, 2, 3, 4]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Caminhos ---
DATA_PATH        = '../1_Preprocessing/dengue_uf_final.csv'
MODELS_DIR       = 'models'
PREDICTIONS_PATH = 'predictions.csv'
SUBMIT_PATH      = 'lstm_climate_afya_percentil_submit.csv'
METRICS_PATH     = 'lstm_climate_afya_percentil_metrics.csv'
PLOTS_PATH       = 'lstm_climate_afya_percentil_plots.pdf'


# ---------------------------------------------------------------------------
# Arquitetura LSTM
# ---------------------------------------------------------------------------
class DengueLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# ---------------------------------------------------------------------------
# Funcoes auxiliares
# ---------------------------------------------------------------------------
def make_sequences(data: np.ndarray, lookback: int):
    """data: (n, n_features) -- coluna 0 e o target."""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i: i + lookback])
        y.append(data[i + lookback, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_exog_future(uf_df, target_col, exog_cols, scaler, n_features):
    """
    Exogenas do periodo de forecast escaladas.
    NaN preenchidos com ffill -> bfill -> 0 (cobre AFYA ausente e clima futuro).
    """
    future_exog = (
        uf_df[uf_df[target_col]][exog_cols]
        .ffill().bfill()
        .fillna(0)
        .values.astype(np.float32)
    )
    h = len(future_exog)
    dummy = np.zeros((h, n_features), dtype=np.float32)
    dummy[:, 1:] = future_exog
    return scaler.transform(dummy)[:, 1:]


def train_model(model, X_train, y_train, epochs, lr, batch_size, patience, device):
    val_size = max(1, int(len(X_train) * 0.10))
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]

    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    best_state = None
    wait = 0

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t).squeeze(), y_val_t).item()
        model.train()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model


def recursive_forecast(model, seed_seq, future_exog_scaled, h, scaler, n_features, device, mc_samples):
    """
    Previsao recursiva com MC Dropout.
    Retorna dicionario com todos os percentis definidos em PERCENTILE_MAP.
    Garante: lower_95 <= lower_90 <= lower_80 <= lower_50 <= pred <= upper_50 <= upper_80 <= upper_90 <= upper_95
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    all_preds = []
    with torch.no_grad():
        for _ in range(mc_samples):
            seq = torch.tensor(seed_seq, dtype=torch.float32).unsqueeze(0).to(device)
            preds = []
            for step in range(h):
                casos_pred = model(seq).item()
                preds.append(casos_pred)
                next_step = np.array(
                    [[casos_pred] + list(future_exog_scaled[step])],
                    dtype=np.float32
                )
                next_t = torch.tensor(next_step, dtype=torch.float32).unsqueeze(0).to(device)
                seq = torch.cat([seq[:, 1:, :], next_t], dim=1)
            all_preds.append(preds)

    all_preds = np.array(all_preds)  # (mc_samples, h)

    def inv_casos(x):
        dummy = np.zeros((len(x), n_features), dtype=np.float32)
        dummy[:, 0] = x
        return scaler.inverse_transform(dummy)[:, 0]

    result = {}
    for col, pct in PERCENTILE_MAP.items():
        result[col] = np.maximum(inv_casos(np.percentile(all_preds, pct, axis=0)), 0)

    return result


def model_path(split, uf):
    return os.path.join(MODELS_DIR, f'lstm_split{split}_{uf}.pt')


def save_model(model, scaler, split, uf, n_features):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({
        'state_dict':  model.state_dict(),
        'scaler':      scaler,
        'input_size':  n_features,
        'hidden_size': HIDDEN_SIZE,
        'num_layers':  NUM_LAYERS,
        'dropout':     DROPOUT,
        'lookback':    LOOKBACK,
        'features':    ALL_FEATURES,
        'split':       split,
        'uf':          uf,
    }, model_path(split, uf))


def load_model(split, uf, device=DEVICE):
    """Reconstroi um modelo salvo por save_model (util para reforecast sem retreinar)."""
    checkpoint = torch.load(model_path(split, uf), map_location=device)
    model = DengueLSTM(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout'],
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint['scaler']


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=['date'])

    missing_cols = [c for c in ALL_FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f'Colunas ausentes: {missing_cols}')

    ufs = sorted(df['uf'].unique())
    print(f'{len(ufs)} UFs | {df["date"].min().date()} -> {df["date"].max().date()}')

    afya_coverage = df['afya_access_count'].notna().mean()
    print(f'Cobertura AFYA: {afya_coverage * 100:.1f}% das linhas (NaN -> 0 no treino)')

    return df, ufs


# ---------------------------------------------------------------------------
# Loop principal: split x UF
# ---------------------------------------------------------------------------
def run_training(df, ufs):
    n_features = len(ALL_FEATURES)
    records = []
    errors = []

    for split in SPLITS:
        train_col = f'train_{split}'
        target_col = f'target_{split}'
        print(f'\n=== Split {split} ===')

        for uf in tqdm(ufs, desc=f'Split {split}'):
            uf_df = df[df['uf'] == uf].sort_values('date').reset_index(drop=True)
            target_rows = uf_df[uf_df[target_col]]
            target_dates = target_rows['date'].values
            target_casos = target_rows['casos'].values
            h = len(target_dates)

            try:
                train_data = (
                    uf_df[uf_df[train_col]][ALL_FEATURES]
                    .ffill().bfill()
                    .fillna(0)
                    .values.astype(np.float32)
                )

                scaler = MinMaxScaler(feature_range=(0, 1))
                train_scaled = scaler.fit_transform(train_data)

                X, y = make_sequences(train_scaled, LOOKBACK)
                if len(X) < 10:
                    raise ValueError(f'Poucos dados: {len(X)} sequencias')

                model = DengueLSTM(
                    input_size=n_features,
                    hidden_size=HIDDEN_SIZE,
                    num_layers=NUM_LAYERS,
                    dropout=DROPOUT
                ).to(DEVICE)
                model = train_model(model, X, y, EPOCHS, LR, BATCH_SIZE, PATIENCE, DEVICE)
                save_model(model, scaler, split, uf, n_features)

                seed_seq = train_scaled[-LOOKBACK:]
                future_exog_scaled = prepare_exog_future(
                    uf_df, target_col, EXOG_FEATURES, scaler, n_features
                )

                forecast = recursive_forecast(
                    model, seed_seq, future_exog_scaled,
                    h, scaler, n_features, DEVICE, MC_SAMPLES
                )

            except Exception as e:
                errors.append({'split': split, 'uf': uf, 'error': str(e)})
                print(f'  ERRO {uf}: {e}')
                continue

            for i in range(h):
                row = {
                    'split': split,
                    'uf': uf,
                    'date': pd.Timestamp(target_dates[i]),
                    'casos_real': target_casos[i],
                }
                for col in PI_COLS:
                    row[col] = forecast[col][i]
                records.append(row)

    results = pd.DataFrame(records)
    print(f'\nTotal de previsoes geradas: {len(results):,}')
    if errors:
        print(pd.DataFrame(errors))

    return results, errors


# ---------------------------------------------------------------------------
# Avaliacao -- splits 1, 2 e 3
# ---------------------------------------------------------------------------
def evaluate(results):
    eval_df = results[results['split'].isin([1, 2, 3])].dropna(subset=['casos_real'])

    metrics = []
    for (split, uf), g in eval_df.groupby(['split', 'uf']):
        mae = mean_absolute_error(g['casos_real'], g['pred'])
        rmse = mean_squared_error(g['casos_real'], g['pred'], squared=False)
        row = {'split': split, 'uf': uf, 'MAE': mae, 'RMSE': rmse}
        for ci in [50, 80, 90, 95]:
            covered = (
                (g['casos_real'] >= g[f'lower_{ci}']) &
                (g['casos_real'] <= g[f'upper_{ci}'])
            ).mean()
            row[f'cov_{ci}'] = covered
        metrics.append(row)

    metrics_df = pd.DataFrame(metrics)

    print('Media por split:')
    print(metrics_df.groupby('split')[['MAE', 'RMSE', 'cov_50', 'cov_80', 'cov_90', 'cov_95']].mean().round(3))
    print('\nMedia geral:')
    print(metrics_df[['MAE', 'RMSE', 'cov_50', 'cov_80', 'cov_90', 'cov_95']].mean().round(3))
    print('\nMAE por UF (media entre splits 1-3):')
    print(
        metrics_df.groupby('uf')['MAE']
        .mean()
        .sort_values(ascending=False)
        .round(1)
        .to_string()
    )

    return metrics_df


# ---------------------------------------------------------------------------
# Visualizacao
# ---------------------------------------------------------------------------
SPLIT_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#9467bd'}

# Bandas do IC mais externo para o mais interno (ordem de fill_between)
CI_BANDS = [
    ('lower_95', 'upper_95', 0.10, 'IC 95%'),
    ('lower_90', 'upper_90', 0.15, 'IC 90%'),
    ('lower_80', 'upper_80', 0.20, 'IC 80%'),
    ('lower_50', 'upper_50', 0.30, 'IC 50%'),
]


def plot_uf(uf, results_df, raw_df, splits_to_plot=None):
    if splits_to_plot is None:
        splits_to_plot = SPLITS

    fig, axes = plt.subplots(
        len(splits_to_plot), 1,
        figsize=(14, 4 * len(splits_to_plot)),
        sharex=False
    )
    if len(splits_to_plot) == 1:
        axes = [axes]

    uf_raw = raw_df[raw_df['uf'] == uf].sort_values('date')

    for ax, split in zip(axes, splits_to_plot):
        fc = results_df[
            (results_df['uf'] == uf) & (results_df['split'] == split)
        ].sort_values('date')

        if fc.empty:
            ax.set_visible(False)
            continue

        forecast_start = fc['date'].min()
        color = SPLIT_COLORS[split]

        ax.plot(uf_raw['date'], uf_raw['casos'],
                color='#555555', lw=1.2, label='Historico')

        for lower_col, upper_col, alpha, label in CI_BANDS:
            ax.fill_between(fc['date'], fc[lower_col], fc[upper_col],
                             color=color, alpha=alpha, label=label)

        ax.plot(fc['date'], fc['pred'],
                color=color, lw=2, label=f'Mediana (split {split})')

        if 'casos_real' in fc.columns and fc['casos_real'].notna().any():
            ax.plot(fc['date'], fc['casos_real'],
                    color='black', lw=1.2, ls='--', label='Real')

        ax.axvline(forecast_start, color='red', ls=':', lw=1, alpha=0.7)
        ax.set_title(f'{uf} — Split {split}', fontsize=11)
        ax.set_ylabel('Casos')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax.legend(fontsize=8, loc='upper left', ncol=2)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'LSTM+Clima+AFYA — {uf} (intervalos de percentil)', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


def make_plots(results, df, ufs):
    with PdfPages(PLOTS_PATH) as pdf:
        for uf in ufs:
            fig = plot_uf(uf, results, df)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f'PDF salvo em {PLOTS_PATH}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print('Device:', DEVICE)
    print(f'Input size: {len(ALL_FEATURES)} features: {ALL_FEATURES}')
    print(f'Percentis gerados: {PERCENTILE_MAP}')

    df, ufs = load_data()
    results, errors = run_training(df, ufs)
    metrics_df = evaluate(results)
    make_plots(results, df, ufs)

    # --- Tabela completa (com casos_real para avaliacao) ---
    results.to_csv(PREDICTIONS_PATH, index=False)

    # --- Tabela de submissao ---
    submit_cols = ['split', 'uf', 'date'] + PI_COLS
    submit = results[submit_cols].copy()
    submit['date'] = submit['date'].dt.strftime('%Y-%m-%d')
    submit.to_csv(SUBMIT_PATH, index=False)

    # --- Metricas ---
    metrics_df.to_csv(METRICS_PATH, index=False)

    print('\nArquivos salvos em 2_model/:')
    print(f'  {PREDICTIONS_PATH}')
    print(f'  {SUBMIT_PATH}')
    print(f'  {METRICS_PATH}')
    print(f'  {PLOTS_PATH}')
    print(f'  {MODELS_DIR}/lstm_split<split>_<uf>.pt  (um arquivo por split x UF)')

    print('\nAmostra da tabela de submissao:')
    print(submit.head(3).to_string(index=False))


if __name__ == '__main__':
    main()
