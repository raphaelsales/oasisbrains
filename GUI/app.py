from nicegui import ui, app
import os, sys, subprocess, psutil, threading, textwrap, gpustat
from pathlib import Path
from datetime import datetime
import time
from queue import Queue, Empty

# =========================
#  PATHS
# =========================
ROOT = Path(__file__).resolve().parent            # /app/alzheimer/GUI
PROJECT_ROOT = ROOT.parent                        # /app/alzheimer
STATIC_DIR = ROOT / 'static'

SCRIPT_QUICK = PROJECT_ROOT / 'quick_analysis.sh'
PY_DATASET   = PROJECT_ROOT / 'dataset_explorer.py'
PY_EARLY     = PROJECT_ROOT / 'alzheimer_early_diagnosis_analysis.py'
PY_MCI       = PROJECT_ROOT / 'mci_clinical_insights.py'
REPORTS_DIR  = PROJECT_ROOT / 'reports'
gpu_stats = gpustat.GPUStatCollection.new_query()

app.add_static_files('/static', str(STATIC_DIR))

# =========================
#  LOG BUS
# =========================
class UiLogBus:
    def __init__(self):
        self._sink = None
        self._q: Queue[str] = Queue()
        self._lock = threading.RLock()
    def attach(self, ui_log):
        with self._lock:
            self._sink = ui_log
        self.flush()
    def push(self, msg: str):
        ts = datetime.now().strftime('%H:%M:%S')
        line = f'[{ts}] {msg}'
        with self._lock:
            if self._sink is not None:
                try:
                    self._sink.push(line)
                except Exception:
                    self._q.put(line)
            else:
                self._q.put(line)
    def flush(self):
        if self._sink is None:
            return
        for _ in range(500):
            try:
                line = self._q.get_nowait()
            except Empty:
                break
            else:
                self._sink.push(line)

LOG = UiLogBus()

# =========================
#  TEE stdout/stderr
# =========================
ORIG_STDOUT = sys.__stdout__
ORIG_STDERR = sys.__stderr__

class TeeToLog:
    def __init__(self, prefix, real_stream):
        self.prefix = prefix
        self.real = real_stream
        self._buf = ""
        self.encoding = getattr(real_stream, 'encoding', 'utf-8')
        self.errors = getattr(real_stream, 'errors', None)
        self.name = getattr(real_stream, 'name', f'<{prefix}>')
    def write(self, s):
        try:
            self.real.write(s); self.real.flush()
        except Exception:
            pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                LOG.push(f'{self.prefix}: {line}')
    def flush(self):
        try:
            self.real.flush()
        except Exception:
            pass
        if self._buf.strip():
            LOG.push(f'{self.prefix}: {self._buf.strip()}')
        self._buf = ""
    def isatty(self): return hasattr(self.real, 'isatty') and self.real.isatty()
    def fileno(self): return self.real.fileno() if hasattr(self.real, 'fileno') else -1
    def writable(self): return hasattr(self.real, 'writable') and self.real.writable()
    def __getattr__(self, item): return getattr(self.real, item)

sys.stdout = TeeToLog("stdout", ORIG_STDOUT)
sys.stderr = TeeToLog("stderr", ORIG_STDERR)

def _excepthook(exc_type, exc, tb):
    import traceback
    LOG.push("EXCEÇÃO NÃO TRATADA:")
    for line in traceback.format_exception(exc_type, exc, tb):
        for sub in line.rstrip().splitlines():
            LOG.push(sub)
sys.excepthook = _excepthook

# =========================
#  Helpers
# =========================
def _ensure_executable(path: Path) -> bool:
    if not path.exists():
        LOG.push(f'ERRO: arquivo não encontrado: {path}'); return False
    if path.suffix == '.sh' and not os.access(path, os.X_OK):
        try:
            os.chmod(path, 0o755); LOG.push(f'permite execução: {path.name}')
        except Exception as e:
            LOG.push(f'ERRO ao dar permissão em {path.name}: {e}'); return False
    return True

def run_stream(cmd: str):
    LOG.push(f'run: {cmd} (cwd={PROJECT_ROOT})')
    try:
        proc = subprocess.Popen(
            cmd, shell=True, cwd=PROJECT_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            LOG.push(line.rstrip())
        code = proc.wait()
        LOG.push(f'fim: processo saiu com código {code}')
    except Exception as e:
        LOG.push(f'ERRO run_stream: {e}')

def run_async(cmd: str):
    threading.Thread(target=run_stream, args=(cmd,), daemon=True).start()

# =========================
#  Actions (9 botões)
# =========================
def act_quick_stats():
    if _ensure_executable(SCRIPT_QUICK): run_async(f'./{SCRIPT_QUICK.name} s')
def act_comprehensive():
    if _ensure_executable(SCRIPT_QUICK): run_async(f'./{SCRIPT_QUICK.name} a')
def act_dataset_explorer():
    if PY_DATASET.exists(): run_async(f'python3 {PY_DATASET.name}')
    else: LOG.push(f'ERRO: {PY_DATASET.name} não existe')
def act_early_dx():
    if PY_EARLY.exists(): run_async(f'python3 {PY_EARLY.name}')
    else: LOG.push(f'ERRO: {PY_EARLY.name} não existe')
def act_mci_clinical():
    if PY_MCI.exists(): run_async(f'python3 {PY_MCI.name}')
    else: LOG.push(f'ERRO: {PY_MCI.name} não existe')
def act_model_performance():
    cmd = textwrap.dedent(r'''
        bash -lc '
        echo "PERFORMANCE DOS MODELOS"
        echo "======================="
        echo
        echo "MODELOS (.h5):"
        ls -lh *.h5 2>/dev/null || echo "  (nenhum .h5)"
        echo
        echo "SCALERS (.joblib):"
        ls -lh *.joblib 2>/dev/null || echo "  (nenhum .joblib)"
        echo
        echo "DATASETS / FIGURAS (.csv/.png):"
        ls -lh *.csv *.png 2>/dev/null || true
        '
    ''').strip()
    run_async(cmd)
def act_tensorboard_status():
    cmd = textwrap.dedent(r'''
        bash -lc '
        echo "TENSORBOARD - STATUS"
        echo "===================="
        if pgrep -f tensorboard >/dev/null; then
            echo "  Rodando em http://localhost:6006"
        else:
            echo "  NÃO está rodando"
            echo "  Iniciar: tensorboard --logdir=logs --host=0.0.0.0 --port=6006 &"
        fi
        echo
        if [ -d logs ]; then
            echo "Tamanho dos logs:"
            du -sh logs 2>/dev/null | sed "s/^/  /"
            echo
            echo "Estrutura (top 10):"
            find logs -type d 2>/dev/null | head -10 | sed "s/^/  /"
        fi
        '
    ''').strip()
    run_async(cmd)
def act_generate_report():
    try:
        REPORTS_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M')
        path = REPORTS_DIR / f'alzheimer_clinical_report_{ts}.txt'
        path.write_text("Relatório clínico gerado automaticamente.\n", encoding='utf-8')
        LOG.push(f'OK: relatório gerado em {path}')
    except Exception as e:
        LOG.push(f'ERRO ao gerar relatório: {e}')
def act_system_info():
    cmd = textwrap.dedent(r'''
        bash -lc '
        echo "INFORMAÇÕES DO SISTEMA"
        echo "======================"
        echo "OS: $(uname -s) $(uname -r)"
        echo "Usuário: $(whoami)"
        echo "Dir: $(pwd)"
        echo "Data/Hora: $(date)"
        echo
        echo "Python:"
        python3 --version
        '
    ''').strip()
    run_async(cmd)

# =========================
#  CSS (arquivo + overrides finais)
# =========================
def inject_css_from_file():
    ts = int(time.time())
    ui.add_head_html('<link rel="preconnect" href="https://fonts.googleapis.com">')
    ui.add_head_html('<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">')
    ui.add_head_html(f'<link rel="stylesheet" href="/static/hud.css?v={ts}">')
    try:
        css_text = (STATIC_DIR / 'hud.css').read_text(encoding='utf-8')
        ui.add_css(css_text)
        LOG.push(f'CSS: hud.css lido ({len(css_text)} bytes)')
    except Exception as e:
        LOG.push(f'CSS: falha ao ler hud.css: {e}')

def inject_final_css():
    ui.add_head_html(r'''
    <style id="hud-final-overrides">
    html, body {
      background: linear-gradient(180deg, #000814, #001d3d) !important;
      color: #e8f9ff !important;
      font-family: 'JetBrains Mono', monospace !important;
      margin: 0 !important; padding: 0 !important;
    }
    .nicegui-content, .q-page, .q-layout { background: transparent !important; }
    .q-card, .nicegui-card {
      background: rgba(10,25,47,.8) !important;
      border: 1px solid rgba(0,229,255,.35) !important;
      border-radius: 18px !important;
      backdrop-filter: saturate(180%) blur(18px) !important;
      box-shadow: 0 24px 80px rgba(0,0,0,.65) !important;
      color: #e8f9ff !important;
    }
    .q-btn, .nicegui-button, button {
      background: radial-gradient(60% 120% at 20% 0%, rgba(0,229,255,.25), rgba(0,229,255,.08)) !important;
      border: 1px solid rgba(0,229,255,.5) !important;
      color: #e8faff !important;
      font-weight: 700 !important; letter-spacing: .06em !important;
      border-radius: 12px !important;
    }
    .q-btn .q-btn__content { background: transparent !important; }
    .q-header, .nicegui-header, header {
      background: linear-gradient(180deg, #0b1220d9 0%, #0b122000 100%) !important;
      border-bottom: 1px solid rgba(0,229,255,.18) !important;
      color: #e8f9ff !important;
    }
    .nicegui-log, [class*="log"] {
      background: rgba(0,0,0,.3) !important; border: 1px solid rgba(0,229,255,.2) !important; color: #e8f9ff !important;
    }
    .q-label, label { color: #e8f9ff !important; }
    .nicegui-row, .row { background: transparent !important; }
    .q-tabs { background: rgba(10,25,47,.8) !important; border-bottom: 1px solid rgba(0,229,255,.35) !important; backdrop-filter: saturate(160%) blur(12px) !important; }
    .q-tab { color: #e8f9ff !important; background: transparent !important; border: 1px solid rgba(0,229,255,.20) !important; margin: 2px !important; border-radius: 8px !important; }
    .q-tab--active { background: rgba(0,229,255,.20) !important; border-color: rgba(0,229,255,.60) !important; color: #00e5ff !important; box-shadow: 0 0 16px rgba(0,229,255,.30) inset !important; }
    .q-tab-panels { background: transparent !important; }
    .hud-plot {
      border: 1px solid rgba(0,229,255,.35) !important;
      border-radius: 14px !important;
      box-shadow: 0 18px 60px rgba(0,0,0,.55) !important;
      background: radial-gradient(120% 160% at 20% 0%, rgba(0,229,255,.06), rgba(0,229,255,.02)) !important;
    }
    </style>
    ''')

# =========================
#  RESULTADOS / GRÁFICOS
# =========================
import json, base64, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

HUD_TEXT = '#e8f9ff'
HUD_PRIMARY = '#00e5ff'
HUD_CYAN_EDGE = (0/255, 229/255, 255/255, 0.35)
HUD_GRID = (232/255, 249/255, 255/255, 0.15)

plt.rcParams.update({
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'savefig.transparent': True,
    'text.color': HUD_TEXT,
    'axes.labelcolor': HUD_TEXT,
    'xtick.color': HUD_TEXT,
    'ytick.color': HUD_TEXT,
    'axes.edgecolor': HUD_CYAN_EDGE,
    'grid.color': HUD_GRID,
    'font.family': 'JetBrains Mono',
})

# fontes de dados de modelo
RESULTS_CANDIDATES = [
    REPORTS_DIR / 'model_results.json',
    PROJECT_ROOT / 'model_results.json',
    REPORTS_DIR / 'metrics.json',
    PROJECT_ROOT / 'metrics.json',
]

# PNGs prontos
EXPLORATORY_CANDIDATES = [
    REPORTS_DIR / 'alzheimer_exploratory_analysis.png',
    PROJECT_ROOT / 'alzheimer_exploratory_analysis.png',
]
FASTSURFER_CANDIDATES = [
    REPORTS_DIR / 'fastsurfer_mci_comprehensive_analysis.png',
    PROJECT_ROOT / 'fastsurfer_mci_comprehensive_analysis.png',
]

# Biomarcadores (csv gerado pelo seu script)
BIOMARKERS_STATS_CANDIDATES = [
    REPORTS_DIR / 'fastsurfer_mci_statistical_analysis.csv',
    PROJECT_ROOT / 'fastsurfer_mci_statistical_analysis.csv',
]
DATASET_CANDIDATES = [
    PROJECT_ROOT / 'alzheimer_complete_dataset.csv',
    REPORTS_DIR / 'alzheimer_complete_dataset.csv',
]

# ---------- Utilidades ----------
def _fig_to_data_url(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight', transparent=True)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{b64}'

def _png_file_to_data_url(path: Path) -> str:
    try:
        b = path.read_bytes()
        b64 = base64.b64encode(b).decode('ascii')
        return f'data:image/png;base64,{b64}'
    except Exception as e:
        LOG.push(f'Gráficos: falha ao ler PNG {path}: {e}')
        fig, ax = plt.subplots(figsize=(5, 3), facecolor='none')
        ax.axis('off'); ax.text(0.5, 0.5, f'Figura não encontrada:\n{path.name}', ha='center', va='center',
                                color=HUD_TEXT, fontsize=12, weight='bold')
        return _fig_to_data_url(fig)

def build_exploratory_image_url() -> str:
    for p in EXPLORATORY_CANDIDATES:
        if p.exists():
            LOG.push(f'Gráficos: usando {p}')
            return _png_file_to_data_url(p)
    LOG.push('Gráficos: exploratory PNG não encontrado; placeholder')
    return _png_file_to_data_url(Path('/dev/null'))

def build_fastsurfer_image_url() -> str:
    for p in FASTSURFER_CANDIDATES:
        if p.exists():
            LOG.push(f'Gráficos: usando {p}')
            return _png_file_to_data_url(p)
    LOG.push('Gráficos: fastsurfer PNG não encontrado; placeholder')
    return _png_file_to_data_url(Path('/dev/null'))

# ---------- Modelo: matriz de confusão / métricas ----------
def _confusion_matrix_from_preds(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    total = len(y_true)
    acc = float((y_true == y_pred).sum()) / total if total else 0.0
    precs, recs, f1s = [], [], []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0
        precs.append(precision); recs.append(recall); f1s.append(f1)
    return {'accuracy': acc, 'precision_macro': float(np.mean(precs)),
            'recall_macro': float(np.mean(recs)), 'f1_macro': float(np.mean(f1s))}

def _load_results_or_dummy():
    for path in RESULTS_CANDIDATES:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                LOG.push(f'Gráficos: carregado {path}')
                return data, True
            except Exception as e:
                LOG.push(f'Gráficos: falha ao ler {path}: {e}')
    rng = np.random.default_rng(7)
    n = 120
    y_true = rng.integers(0, 3, size=n).tolist()
    noise  = (rng.random(n) < 0.15)
    y_pred = np.where(noise, (np.array(y_true) + 1) % 3, y_true).tolist()
    labels = ['CN','MCI','AD']
    data = {'y_true': y_true, 'y_pred': y_pred, 'labels': labels, 'metrics': None}
    LOG.push('Gráficos: usando dados de exemplo (model_results.json não encontrado)')
    return data, False

def render_confusion_matrix_png(y_true, y_pred, labels):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(labels)
    cm = _confusion_matrix_from_preds(y_true, y_pred, n)
    vmax = cm.max() if cm.size else 1
    fig, ax = plt.subplots(figsize=(5, 4), facecolor='none')
    im = ax.imshow(cm, cmap='viridis')
    ax.set_title('Matriz de Confusão', color=HUD_TEXT)
    ax.set_xlabel('Predito', color=HUD_TEXT); ax.set_ylabel('Verdadeiro', color=HUD_TEXT)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, color=HUD_TEXT); ax.set_yticklabels(labels, color=HUD_TEXT)
    for s in ax.spines.values(): s.set_edgecolor(HUD_CYAN_EDGE)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color=HUD_TEXT if cm[i, j] > vmax/2 else '#0a162f', fontsize=10, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor(HUD_CYAN_EDGE); cbar.ax.yaxis.set_tick_params(color=HUD_TEXT)
    plt.setp(cbar.ax.get_yticklabels(), color=HUD_TEXT)
    return _fig_to_data_url(fig)

def render_metrics_bar_png(metrics_dict: dict):
    if not metrics_dict:
        data, _ = _load_results_or_dummy()
        y_true, y_pred = np.array(data['y_true']), np.array(data['y_pred'])
        n = len(data.get('labels', [])) or int(max(y_true.max(), y_pred.max())+1)
        metrics_dict = _safe_metrics(y_true, y_pred, n)
    keys = ['accuracy','precision_macro','recall_macro','f1_macro']
    vals = [metrics_dict.get(k, 0.0) for k in keys]
    fig, ax = plt.subplots(figsize=(5, 3), facecolor='none')
    ax.bar(keys, vals)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score', color=HUD_TEXT); ax.set_title('Métricas do Modelo', color=HUD_TEXT)
    ax.tick_params(colors=HUD_TEXT)
    for s in ax.spines.values(): s.set_edgecolor(HUD_CYAN_EDGE)
    for i, v in enumerate(vals):
        ax.text(i, min(v + 0.03, 1.0), f'{v:.2f}', ha='center', va='bottom', fontsize=10, color=HUD_TEXT, fontweight='bold')
    return _fig_to_data_url(fig)

def build_results_images():
    data, _ = _load_results_or_dummy()
    y_true = data.get('y_true', [])
    y_pred = data.get('y_pred', [])
    labels = data.get('labels') or [str(c) for c in sorted(set(list(y_true)+list(y_pred)))]
    return render_confusion_matrix_png(y_true, y_pred, labels), render_metrics_bar_png(data.get('metrics'))

# ---------- Biomarcadores: carregamento ----------
def _load_biomarkers_table() -> pd.DataFrame:
    for p in BIOMARKERS_STATS_CANDIDATES:
        if p.exists():
            try:
                df = pd.read_csv(p)
                required = {'feature','anatomical_region','percent_change','p_value_ttest','cohens_d'}
                if required.issubset(df.columns):
                    LOG.push(f'Biomarcadores: tabela carregada {p}')
                    return df
                else:
                    LOG.push(f'Biomarcadores: CSV {p} sem colunas necessárias {required}')
            except Exception as e:
                LOG.push(f'Biomarcadores: erro ao ler {p}: {e}')
    # dummy
    rng = np.random.default_rng(42)
    feats = [f'left_hippocampus_{i}' for i in range(4)] + [f'left_amygdala_{i}' for i in range(3)] + [f'entorhinal_{i}' for i in range(3)]
    regions = (['Hipocampo']*4)+(['Amígdala']*3)+(['Córtex Entorrinal']*3)
    df = pd.DataFrame({
        'feature': feats,
        'anatomical_region': regions,
        'percent_change': rng.normal(-0.4, 0.2, size=len(feats))*100,
        'p_value_ttest': np.clip(rng.uniform(0,0.2,size=len(feats)), 1e-6, None),
        'cohens_d': np.clip(rng.normal(0.35, 0.15, size=len(feats)), 0, None),
    })
    LOG.push('Biomarcadores: usando dados dummy (CSV não encontrado)')
    return df

def _load_mci_dataset_for_raw() -> pd.DataFrame|None:
    for p in DATASET_CANDIDATES:
        if p.exists():
            try:
                df = pd.read_csv(p)
                if 'cdr' in df.columns:
                    mci_df = df[df['cdr'].isin([0.0,0.5])].copy()
                    LOG.push(f'Biomarcadores: dataset bruto carregado {p} (MCI vs Normal)')
                    return mci_df
            except Exception as e:
                LOG.push(f'Biomarcadores: erro ao ler dataset {p}: {e}')
    LOG.push('Biomarcadores: dataset bruto não encontrado; alguns gráficos usarão placeholder/dummy')
    return None

# ---------- Biomarcadores: figuras individuais ----------
def render_bio_top10_effect_png(bio_df: pd.DataFrame):
    top = bio_df.sort_values(['cohens_d','p_value_ttest'], ascending=[False, True]).head(10)
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    y = np.arange(len(top))
    colors = ['#ff7675' if p < 0.05 else '#f8c291' for p in top['p_value_ttest']]
    ax.barh(y, top['cohens_d'], color=colors)
    ax.set_yticks(y); ax.set_yticklabels([f.replace('_',' ')[:28] for f in top['feature']], color=HUD_TEXT)
    ax.set_xlabel("Cohen's d", color=HUD_TEXT); ax.set_title('Top 10 por Effect Size', color=HUD_TEXT)
    ax.axvline(0.5, ls='--', alpha=.5, color=HUD_TEXT); ax.axvline(0.8, ls='--', alpha=.6, color=HUD_TEXT)
    for s in ax.spines.values(): s.set_edgecolor(HUD_CYAN_EDGE)
    return _fig_to_data_url(fig)

def render_bio_pvalue_hist_png(bio_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    ax.hist(bio_df['p_value_ttest'].values, bins=20, alpha=.8)
    ax.axvline(0.05, color='#ff5e57', ls='--', label='p=0.05'); ax.axvline(0.01, color='#c0392b', ls='--', label='p=0.01')
    ax.set_xlabel('p-value', color=HUD_TEXT); ax.set_ylabel('Frequência', color=HUD_TEXT)
    ax.set_title('Distribuição de p-values', color=HUD_TEXT); ax.legend()
    for s in ax.spines.values(): s.set_edgecolor(HUD_CYAN_EDGE)
    return _fig_to_data_url(fig)

def render_bio_effect_by_region_png(bio_df: pd.DataFrame):
    reg = bio_df.groupby('anatomical_region')['cohens_d'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    ax.bar(reg.index, reg.values)
    ax.set_ylabel("Cohen's d médio", color=HUD_TEXT); ax.set_title('Effect Size por Região', color=HUD_TEXT)
    ax.tick_params(axis='x', rotation=20, colors=HUD_TEXT)
    for s in ax.spines.values(): s.set_edgecolor(HUD_CYAN_EDGE)
    return _fig_to_data_url(fig)

def render_bio_volcano_png(bio_df: pd.DataFrame):
    x = -np.log10(bio_df['p_value_ttest'].values + 1e-10)
    y = bio_df['cohens_d'].values
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    colors = np.where((bio_df['p_value_ttest']<0.05)&(bio_df['cohens_d']>0.3),'#ff7675',
              np.where(bio_df['p_value_ttest']<0.05,'#f8c291','#95a5a6'))
    ax.scatter(x, y, c=colors, alpha=.75)
    ax.axhline(0.3, ls='--', alpha=.5, color=HUD_TEXT); ax.axvline(-np.log10(0.05), ls='--', alpha=.5, color=HUD_TEXT)
    ax.set_xlabel('-log10(p-value)', color=HUD_TEXT); ax.set_ylabel("Cohen's d", color=HUD_TEXT)
    ax.set_title('Volcano: Significância x Effect', color=HUD_TEXT)
    for s in ax.spines.values(): s.set_edgecolor(HUD_CYAN_EDGE)
    return _fig_to_data_url(fig)

def render_bio_distribution_top_feature_png(bio_df: pd.DataFrame, raw_df: pd.DataFrame|None):
    feat = bio_df.sort_values(['cohens_d','p_value_ttest'], ascending=[False, True]).iloc[0]['feature']
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    if raw_df is not None and feat in raw_df.columns and 'cdr' in raw_df.columns:
        normal = raw_df[raw_df['cdr']==0.0][feat].dropna()
        mci    = raw_df[raw_df['cdr']==0.5][feat].dropna()
        ax.hist([normal, mci], bins=20, alpha=.75, label=['Normal','MCI'])
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Dataset bruto não encontrado\n(ou feature ausente)', ha='center', va='center', color=HUD_TEXT)
        ax.axis('off')
        return _fig_to_data_url(fig)
    ax.set_xlabel(feat.replace('_',' ')[:28], color=HUD_TEXT); ax.set_ylabel('Frequência', color=HUD_TEXT)
    ax.set_title(f'Distribuição: {feat}', color=HUD_TEXT)
    for s in ax.spines.values(): s.set_edgecolor(HUD_CYAN_EDGE)
    return _fig_to_data_url(fig)

def render_bio_corr_heat_png(bio_df: pd.DataFrame, raw_df: pd.DataFrame|None):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='none')
    if raw_df is not None:
        top_feats = [f for f in bio_df.sort_values(['cohens_d','p_value_ttest'], ascending=[False, True]).head(8)['feature'] if f in raw_df.columns]
        if len(top_feats) >= 2:
            corr = raw_df[top_feats].corr()
            sns.heatmap(corr, ax=ax, cmap='coolwarm', center=0, annot=True, fmt='.2f',
                        xticklabels=[t[:10] for t in top_feats], yticklabels=[t[:10] for t in top_feats],
                        cbar_kws={'shrink': .8})
            ax.set_title('Correlação entre Top Biomarcadores', color=HUD_TEXT)
            ax.tick_params(colors=HUD_TEXT)
            for t in ax.texts: t.set_color(HUD_TEXT)
            return _fig_to_data_url(fig)
    ax.axis('off'); ax.text(0.5,0.5,'Sem dados brutos suficientes\npara correlação', ha='center', va='center', color=HUD_TEXT)
    return _fig_to_data_url(fig)

def render_bio_box_by_region_png(bio_df: pd.DataFrame, raw_df: pd.DataFrame|None):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    if raw_df is None:
        ax.axis('off'); ax.text(0.5,0.5,'Dataset bruto não encontrado', ha='center', va='center', color=HUD_TEXT)
        return _fig_to_data_url(fig)
    regions = ['Hipocampo','Córtex Entorrinal','Amígdala']
    data = []; labels = []
    for region in regions:
        feats = bio_df[bio_df['anatomical_region']==region]['feature'].tolist()
        feats = [f for f in feats if f in raw_df.columns]
        if not feats: continue
        feat = feats[0]
        normal = raw_df[raw_df['cdr']==0.0][feat].dropna().tolist()
        mci    = raw_df[raw_df['cdr']==0.5][feat].dropna().tolist()
        if normal and mci:
            data.extend([normal, mci]); labels.extend([f'{region}\nNormal', f'{region}\nMCI'])
    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        cols = ['#74b9ff','#ff7675']*(len(labels)//2)
        for patch, c in zip(bp['boxes'], cols): patch.set_facecolor(c)
        ax.set_title('Comparação por Região (exemplo de feature)', color=HUD_TEXT)
        ax.tick_params(axis='x', rotation=15, colors=HUD_TEXT)
        for s in ax.spines.values(): s.set_edgecolor(HUD_CYAN_EDGE)
    else:
        ax.axis('off'); ax.text(0.5,0.5,'Sem features válidas por região', ha='center', va='center', color=HUD_TEXT)
    return _fig_to_data_url(fig)

def render_bio_percent_change_by_region_png(bio_df: pd.DataFrame):
    reg = bio_df.groupby('anatomical_region')['percent_change'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    colors = ['#ff7675' if v<0 else '#55efc4' for v in reg.values]
    ax.bar(reg.index, reg.values, color=colors)
    ax.axhline(0, color=HUD_TEXT, alpha=.3)
    ax.set_ylabel('Mudança média (%)', color=HUD_TEXT); ax.set_title('Mudança Percentual por Região', color=HUD_TEXT)
    ax.tick_params(axis='x', rotation=20, colors=HUD_TEXT)
    for s in ax.spines.values(): s.set_edgecolor(HUD_CYAN_EDGE)
    return _fig_to_data_url(fig)

def render_bio_summary_png(bio_df: pd.DataFrame):
    n_total = len(bio_df)
    n_sig = int((bio_df['p_value_ttest']<0.05).sum())
    n_p001 = int((bio_df['p_value_ttest']<0.01).sum())
    n_large = int((bio_df['cohens_d']>0.8).sum())
    n_medium = int((bio_df['cohens_d']>0.5).sum())
    top = bio_df.sort_values(['cohens_d','p_value_ttest'], ascending=[False, True]).head(1).iloc[0]
    lines = [
        'RESUMO ESTATÍSTICO','','Total de métricas analisadas: {}'.format(n_total),'',
        'Significância:','• p < 0.05: {}'.format(n_sig),'• p < 0.01: {}'.format(n_p001),'',
        'Effect size:','• Large (>0.8): {}'.format(n_large),'• Medium (>0.5): {}'.format(n_medium),
        '• Small (>0.2): {}'.format(int((bio_df["cohens_d"]>0.2).sum())),'',
        'Mais discriminativo:','• {}'.format(top['feature']),
        '• Effect size: {:.3f}'.format(float(top['cohens_d'])),
        '• p-value: {:.4f}'.format(float(top['p_value_ttest'])),
    ]
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    ax.axis('off')
    ax.text(0.02, 0.98, '\n'.join(lines), va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=.85))
    return _fig_to_data_url(fig)

# =========================
#  Tema
# =========================
ui.dark_mode()
inject_css_from_file()

# =========================
#  UI PAGE
# =========================
@ui.page('/')

def main():
    inject_final_css()

    with ui.header().classes('hud-header bg-transparent'):
        ui.label('ALZHEIMER ANALYSIS SUITE').classes('hud-title-xl')
        ui.label('Detecção precoce • MCI • Biomarcadores').classes('hud-subtle')

    with ui.row().classes('hud-row'):
        with ui.card().classes('hud-card hud-glass hud-depth'):
            ui.label('Visão Geral').classes('hud-title')
            g = ui.label().classes('hud-subtle')
            def tick():
                g.set_text(
                    f'CPU {psutil.cpu_percent():.0f}% • '
                    f'RAM {psutil.virtual_memory().percent:.0f}% • '
                    f'GPU {gpu_stats.gpus[0].utilization:.0f}% • '
                    f'Disco {psutil.disk_usage("/").percent:.0f}% • '
                    f'{time.strftime("%H:%M:%S")}'
                )
            ui.timer(2.0, tick)

        with ui.card().classes('hud-card hud-glass w-[860px]'):
            ui.label('Ações').classes('hud-title')
            with ui.row().classes('gap-3 flex-wrap'):
                ui.button('1) Estatísticas básicas', on_click=act_quick_stats).classes('hud-btn')
                ui.button('2) Análise abrangente', on_click=act_comprehensive).classes('hud-btn')
                ui.button('3) Explorador do dataset', on_click=act_dataset_explorer).classes('hud-btn')
                ui.button('4) Diagnóstico precoce', on_click=act_early_dx).classes('hud-btn')
                ui.button('5) Análise clínica CCL', on_click=act_mci_clinical).classes('hud-btn')
                ui.button('6) Performance dos modelos', on_click=act_model_performance).classes('hud-btn')
                ui.button('7) Status TensorBoard', on_click=act_tensorboard_status).classes('hud-btn hud-btn--warn')
                ui.button('8) Gerar relatório clínico', on_click=act_generate_report).classes('hud-btn hud-btn--ok')
                ui.button('9) Informações do sistema', on_click=act_system_info).classes('hud-btn')

        # ===== Visualização de Resultados =====
        with ui.card().classes('hud-card hud-glass w-[860px]'):
            ui.label('Visualização de Resultados').classes('hud-title')

            tabs = ui.tabs().classes('w-full')
            with tabs:
                ui.tab('Matriz de Confusão')
                ui.tab('Métricas')
                ui.tab('Exploratória')
                ui.tab('Biomarcadores MCI')
                ui.tab('Biomarcadores (ind.)')

            with ui.tab_panels(tabs, value='Matriz de Confusão').classes('w-full'):
                with ui.tab_panel('Matriz de Confusão'):
                    cm_img = ui.image().classes('w-full rounded-lg hud-plot')
                with ui.tab_panel('Métricas'):
                    m_img = ui.image().classes('w-full rounded-lg hud-plot')
                with ui.tab_panel('Exploratória'):
                    exp_img = ui.image().classes('w-full rounded-lg hud-plot')
                with ui.tab_panel('Biomarcadores MCI'):
                    fast_img = ui.image().classes('w-full rounded-lg hud-plot')
                with ui.tab_panel('Biomarcadores (ind.)'):
                    # grid simples: 2 por linha
                    with ui.row().classes('gap-3 flex-wrap'):
                        b_top10   = ui.image().classes('w-[410px] rounded-lg hud-plot')
                        b_pval    = ui.image().classes('w-[410px] rounded-lg hud-plot')
                        b_region  = ui.image().classes('w-[410px] rounded-lg hud-plot')
                        b_volcano = ui.image().classes('w-[410px] rounded-lg hud-plot')
                        b_dist    = ui.image().classes('w-[410px] rounded-lg hud-plot')
                        b_corr    = ui.image().classes('w-[410px] rounded-lg hud-plot')
                        b_box     = ui.image().classes('w-[410px] rounded-lg hud-plot')
                        b_pct     = ui.image().classes('w-[410px] rounded-lg hud-plot')
                        b_sum     = ui.image().classes('w-[840px] rounded-lg hud-plot')

            def _refresh_plots():
                try:
                    # modelo
                    cm_url, metrics_url = build_results_images()
                    cm_img.set_source(cm_url); m_img.set_source(metrics_url)
                    # PNGs prontos
                    exp_img.set_source(build_exploratory_image_url())
                    fast_img.set_source(build_fastsurfer_image_url())
                    # biomarcadores individuais
                    bio_df = _load_biomarkers_table()
                    raw_df = _load_mci_dataset_for_raw()
                    b_top10.set_source(  render_bio_top10_effect_png(bio_df) )
                    b_pval.set_source(   render_bio_pvalue_hist_png(bio_df) )
                    b_region.set_source( render_bio_effect_by_region_png(bio_df) )
                    b_volcano.set_source(render_bio_volcano_png(bio_df) )
                    b_dist.set_source(   render_bio_distribution_top_feature_png(bio_df, raw_df) )
                    b_corr.set_source(   render_bio_corr_heat_png(bio_df, raw_df) )
                    b_box.set_source(    render_bio_box_by_region_png(bio_df, raw_df) )
                    b_pct.set_source(    render_bio_percent_change_by_region_png(bio_df) )
                    b_sum.set_source(    render_bio_summary_png(bio_df) )
                    LOG.push('Gráficos: atualizados')
                except Exception as e:
                    LOG.push(f'Gráficos: erro ao atualizar: {e}')

            with ui.row().classes('gap-3 mt-2'):
                ui.button('Atualizar gráficos', on_click=_refresh_plots).classes('hud-btn hud-btn--ok')

            _refresh_plots()

        with ui.card().classes('hud-card hud-glass'):
            ui.label('Console Principal').classes('hud-title')
            ui_log = ui.log(max_lines=1000).classes('hud-log')
            LOG.attach(ui_log)
            ui.timer(0.5, LOG.flush)
            LOG.push('— aplicação iniciada —')
            LOG.push(f'ROOT={ROOT}')
            LOG.push(f'PROJECT_ROOT={PROJECT_ROOT}')
            for p in [SCRIPT_QUICK, PY_DATASET, PY_EARLY, PY_MCI]:
                LOG.push(f'check: {p.name} -> {"OK" if p.exists() else "NÃO ENCONTRADO"}')

# =========================
#  STARTUP
# =========================
def start_server(port: int) -> None:
    LOG.push(f'tentando iniciar servidor na porta {port}...')
    ui.run(title='Alzheimer HUD', host='0.0.0.0', port=port, reload=False, show=False)

if __name__ == '__main__':
    try:
        try:
            start_server(8080)
        except OSError as e:
            LOG.push(f'porta 8080 falhou ({e}); tentando 7860...')
            start_server(7860)
    except Exception as e:
        import traceback
        tb = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        print('[FATAL] falha ao iniciar NiceGUI:', e); print(tb)
        LOG.push('[FATAL] falha ao iniciar NiceGUI')
        for line in tb.splitlines(): LOG.push(line)
        time.sleep(3)
