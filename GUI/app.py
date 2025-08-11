from nicegui import ui, app
import os, sys, subprocess, psutil, threading, textwrap
from pathlib import Path
from datetime import datetime
import time
from queue import Queue, Empty

# =========================
#  PATHS
# =========================
ROOT = Path(__file__).resolve().parent            # /app/alzheimer/GUI
PROJECT_ROOT = ROOT.parent                         # /app/alzheimer
STATIC_DIR = ROOT / 'static'

SCRIPT_QUICK = PROJECT_ROOT / 'quick_analysis.sh'
PY_DATASET   = PROJECT_ROOT / 'dataset_explorer.py'
PY_EARLY     = PROJECT_ROOT / 'alzheimer_early_diagnosis_analysis.py'
PY_MCI       = PROJECT_ROOT / 'mci_clinical_insights.py'
REPORTS_DIR  = PROJECT_ROOT / 'reports'

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
    """Carrega fontes + hud.css (arquivo)"""
    ts = int(time.time())
    ui.add_head_html('<link rel="preconnect" href="https://fonts.googleapis.com">')
    ui.add_head_html('<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">')
    ui.add_head_html(f'<link rel="stylesheet" href="/static/hud.css?v={ts}">')
    try:
        css_text = (STATIC_DIR / 'hud.css').read_text(encoding='utf-8')
        ui.add_css(css_text)  # inline (reforço)
        LOG.push(f'CSS: hud.css lido ({len(css_text)} bytes)')
    except Exception as e:
        LOG.push(f'CSS: falha ao ler hud.css: {e}')

def inject_final_css():
    """Overrides finais SEM seletor universal; inserido **por último** dentro da página."""
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
      display: inline-flex !important; visibility: visible !important; opacity: 1 !important;
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
    /* Tabs */
    .q-tabs { background: rgba(10,25,47,.8) !important; border-bottom: 1px solid rgba(0,229,255,.35) !important; backdrop-filter: saturate(160%) blur(12px) !important; }
    .q-tab { color: #e8f9ff !important; background: transparent !important; border: 1px solid rgba(0,229,255,.20) !important; margin: 2px !important; border-radius: 8px !important; }
    .q-tab--active { background: rgba(0,229,255,.20) !important; border-color: rgba(0,229,255,.60) !important; color: #00e5ff !important; box-shadow: 0 0 16px rgba(0,229,255,.30) inset !important; }
    .q-tab-panels { background: transparent !important; }
    </style>
    ''')

# =========================
#  RESULTADOS / GRÁFICOS
# =========================
import json, base64, io
import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend sem janela
import matplotlib.pyplot as plt

RESULTS_CANDIDATES = [
    REPORTS_DIR / 'model_results.json',
    PROJECT_ROOT / 'model_results.json',
    REPORTS_DIR / 'metrics.json',
    PROJECT_ROOT / 'metrics.json',
]

def _fig_to_data_url(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{b64}'

def _confusion_matrix_from_preds(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    total = len(y_true)
    acc = float((y_true == y_pred).sum()) / total if total else 0.0
    # precisão/recall macro simples (evita sklearn)
    precs, recs, f1s = [], [], []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0
        precs.append(precision); recs.append(recall); f1s.append(f1)
    return {
        'accuracy': acc,
        'precision_macro': float(np.mean(precs)),
        'recall_macro': float(np.mean(recs)),
        'f1_macro': float(np.mean(f1s)),
    }

def _load_results_or_dummy():
    """Tenta carregar resultados reais; se não achar, gera exemplo e loga aviso."""
    for path in RESULTS_CANDIDATES:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                LOG.push(f'Gráficos: carregado {path}')
                return data, True
            except Exception as e:
                LOG.push(f'Gráficos: falha ao ler {path}: {e}')
    # dummy
    rng = np.random.default_rng(7)
    n = 120
    y_true = rng.integers(0, 3, size=n).tolist()
    noise  = (rng.random(n) < 0.15)
    y_pred = np.where(noise, (np.array(y_true) + 1) % 3, y_true).tolist()
    labels = ['CN','MCI','AD']
    data = {
        'y_true': y_true,
        'y_pred': y_pred,
        'labels': labels,
        'metrics': None,
    }
    LOG.push('Gráficos: usando dados de exemplo (model_results.json não encontrado)')
    return data, False

def render_confusion_matrix_png(y_true, y_pred, labels):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(labels)
    cm = _confusion_matrix_from_preds(y_true, y_pred, n)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='viridis')
    ax.set_title('Matriz de Confusão')
    ax.set_xlabel('Predito'); ax.set_ylabel('Verdadeiro')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    # anotações
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_to_data_url(fig)

def render_metrics_bar_png(metrics_dict: dict):
    if not metrics_dict:
        # se não veio, calcula básico a partir de dummy loader
        data, _ = _load_results_or_dummy()
        y_true, y_pred = np.array(data['y_true']), np.array(data['y_pred'])
        n = len(data.get('labels', [])) or int(max(y_true.max(), y_pred.max())+1)
        metrics_dict = _safe_metrics(y_true, y_pred, n)

    keys = ['accuracy','precision_macro','recall_macro','f1_macro']
    vals = [metrics_dict.get(k, 0.0) for k in keys]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(keys, vals)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Métricas do Modelo')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    return _fig_to_data_url(fig)

def build_results_images():
    """Retorna (data_url_confusion_matrix, data_url_metrics)."""
    data, _ = _load_results_or_dummy()
    y_true = data.get('y_true', [])
    y_pred = data.get('y_pred', [])
    labels = data.get('labels', None)
    # se labels não existir, infere
    if not labels:
        classes = sorted(set(list(y_true) + list(y_pred)))
        labels = [str(c) for c in classes]
    cm_url = render_confusion_matrix_png(y_true, y_pred, labels)
    metrics_url = render_metrics_bar_png(data.get('metrics'))
    return cm_url, metrics_url

# =========================
#  Tema
# =========================
ui.dark_mode()
# ui.colors(...) é opcional; se quiser manter, deixe.
# ui.colors(primary='#00e5ff', secondary='#ffd60a', positive='#38ff8c', dark='#000814')
inject_css_from_file()

# =========================
#  UI PAGE
# =========================
@ui.page('/')
def main():
    # injeta overrides **dentro** da página para ficar por último
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
                ui.button('5) Análise clínica MCI', on_click=act_mci_clinical).classes('hud-btn')
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

            with ui.tab_panels(tabs, value='Matriz de Confusão').classes('w-full'):
                with ui.tab_panel('Matriz de Confusão'):
                    cm_img = ui.image().classes('w-full rounded-lg')
                with ui.tab_panel('Métricas'):
                    m_img = ui.image().classes('w-full rounded-lg')

            def _refresh_plots():
                try:
                    cm_url, metrics_url = build_results_images()
                    cm_img.set_source(cm_url)
                    m_img.set_source(metrics_url)
                    LOG.push('Gráficos: atualizados')
                except Exception as e:
                    LOG.push(f'Gráficos: erro ao atualizar: {e}')

            with ui.row().classes('gap-3 mt-2'):
                ui.button('Atualizar gráficos', on_click=_refresh_plots).classes('hud-btn hud-btn--ok')

            # carrega uma vez ao abrir a página
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