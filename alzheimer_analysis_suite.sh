from nicegui import ui, app
import subprocess, psutil, time, threading, os
from pathlib import Path

# === paths ===
ROOT = Path(__file__).resolve().parent            # /app/alzheimer/GUI
PROJECT_ROOT = ROOT.parent                         # /app/alzheimer
SCRIPT_QUICK = PROJECT_ROOT / 'quick_analysis.sh'  # /app/alzheimer/quick_analysis.sh

# === static ===
app.add_static_files('/static', str(ROOT / 'static'))

# === logging helper ===
log = None
def push(msg: str):
    if log is not None:
        log.push(msg)

# === stream runner ===
def run_stream(cmd: str, workdir: Path):
    try:
        proc = subprocess.Popen(
            cmd, shell=True, cwd=str(workdir), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        for line in proc.stdout:
            push(line.rstrip())
        proc.wait()
        push(f'[fim] processo saiu com código {proc.returncode}')
    except Exception as e:
        push(f'[erro] {e}')

def run_async(cmd: str, workdir: Path = PROJECT_ROOT):
    push(f'[run] {cmd} (cwd={workdir})')
    threading.Thread(target=run_stream, args=(cmd, workdir), daemon=True).start()

# === util: garante executável do quick_analysis.sh ===
def ensure_quick_script():
    if not SCRIPT_QUICK.exists():
        push(f'[erro] script não encontrado: {SCRIPT_QUICK}')
        return False
    if not os.access(SCRIPT_QUICK, os.X_OK):
        try:
            os.chmod(SCRIPT_QUICK, 0o755)
            push('[info] permissões ajustadas em quick_analysis.sh (chmod +x)')
        except Exception as e:
            push(f'[erro] não consegui tornar executável: {e}')
            return False
    return True

# === head ===
ui.add_head_html('<link rel="preconnect" href="https://fonts.googleapis.com">')
ui.add_head_html('<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">')
ui.add_head_html('<link rel="stylesheet" href="/static/hud.css">')

# === UI ===
@ui.page('/')
def main():
    with ui.header().classes('hud-header'):
        ui.label('ALZHEIMER ANALYSIS SUITE').classes('hud-title-xl')
        ui.label('Detecção precoce • MCI • Biomarcadores').classes('hud-subtle')

    with ui.row().classes('hud-row'):
        # Visão Geral
        with ui.card().classes('hud-card hud-glass hud-depth'):
            ui.label('Visão Geral').classes('hud-title')
            sys_label = ui.label().classes('hud-subtle')

            def _tick():
                sys_label.set_text(
                    f'CPU {psutil.cpu_percent():.0f}% • '
                    f'RAM {psutil.virtual_memory().percent:.0f}% • '
                    f'{time.strftime("%H:%M:%S")}'
                )

            ui.timer(2.0, _tick)

        # Ações – 9 opções do Suite
        with ui.card().classes('hud-card hud-glass grow'):
            ui.label('Ações (Suite Completa)').classes('hud-title')

            with ui.grid(columns=3).classes('gap-3'):
                # 1. Estatísticas básicas
                def act_quick_stats():
                    if ensure_quick_script():
                        run_async(f'{SCRIPT_QUICK} s')
                ui.button('1) Estatísticas básicas', on_click=act_quick_stats).classes('hud-btn')

                # 2. Análise abrangente
                def act_comprehensive():
                    if ensure_quick_script():
                        run_async(f'{SCRIPT_QUICK} a')
                ui.button('2) Análise abrangente', on_click=act_comprehensive).classes('hud-btn')

                # 3. Explorador do dataset (completo)
                ui.button(
                    '3) Explorador do dataset',
                    on_click=lambda: run_async('python3 dataset_explorer.py')
                ).classes('hud-btn')

                # 4. Diagnóstico precoce (completo)
                ui.button(
                    '4) Diagnóstico precoce',
                    on_click=lambda: run_async('python3 alzheimer_early_diagnosis_analysis.py')
                ).classes('hud-btn')

                # 5. Análise clínica MCI
                ui.button(
                    '5) Análise clínica MCI',
                    on_click=lambda: run_async('python3 mci_clinical_insights.py')
                ).classes('hud-btn')

                # 6. Performance dos modelos
                PERF_CMD = r'''bash -lc '
echo "PERFORMANCE DOS MODELOS"
echo "======================="
echo
echo "MODELOS TREINADOS:"
ls -lh *.h5 2>/dev/null | awk "{printf \"  %s (%s)\n\", \$9, \$5}"
echo
echo "SCALERS E PREPROCESSADORES:"
ls -lh *.joblib 2>/dev/null | awk "{printf \"  %s (%s)\n\", \$9, \$5}"
echo
echo "DATASET E VISUALIZACOES:"
ls -lh *.csv *.png 2>/dev/null | awk "{printf \"  %s (%s)\n\", \$9, \$5}"
echo
echo "GPU PERFORMANCE (exemplo esperado):"
echo "  Placa: NVIDIA RTX A4000"
echo "  Mixed Precision: Ativada"
echo "  Speedup: ~6-10x vs CPU"
' '''
                ui.button('6) Performance dos modelos',
                          on_click=lambda: run_async(PERF_CMD)).classes('hud-btn')

                # 7. Status TensorBoard
                TENSOR_CMD = r'''bash -lc '
echo "TENSORBOARD - MONITORAMENTO"
echo "==========================="
if pgrep -f tensorboard > /dev/null; then
  echo "  TensorBoard está RODANDO"
  echo "  Acesso: http://localhost:6006"
  if [ -d logs ]; then
    echo
    echo "Tamanho dos logs: $(du -sh logs 2>/dev/null | cut -f1)"
    echo
    echo "Algumas pastas:"
    find logs -type d 2>/dev/null | head -10 | sed "s/^/  /"
  fi
else
  echo "  TensorBoard NÃO está rodando"
  echo "  Inicie com:"
  echo "    tensorboard --logdir=logs --host=0.0.0.0 --port=6006 &"
fi
' '''
                ui.button('7) Status TensorBoard',
                          on_click=lambda: run_async(TENSOR_CMD)).classes('hud-btn')

                # 8. Gerar relatório clínico
                REPORT_CMD = r'''bash -lc '
report_file="alzheimer_clinical_report_$(date +%Y%m%d_%H%M).txt"
cat > "$report_file" << "EOF"
RELATORIO CLINICO - DIAGNOSTICO PRECOCE DE ALZHEIMER
====================================================
(Resumo gerado automaticamente)
- Total de sujeitos: 405
- MCI e biomarcadores conforme análises do projeto

(Edite o script para personalizar completamente o conteúdo)
EOF
echo "Relatório gerado: $report_file"
[ -f "$report_file" ] && tail -n +1 "$report_file"
' '''
                ui.button('8) Gerar relatório clínico',
                          on_click=lambda: run_async(REPORT_CMD)).classes('hud-btn')

                # 9. Informações do sistema
                SYS_CMD = r'''bash -lc '
echo "INFORMAÇÕES DO SISTEMA"
echo "======================"
echo "OS: $(uname -s) $(uname -r)"
echo "Usuário: $(whoami)"
echo "Diretório: $(pwd)"
echo "Data/Hora: $(date)"
echo
echo "PYTHON:"
python3 --version
echo
echo "PACOTES PRINCIPAIS:"
python3 - <<PY
packages = ["pandas","numpy","tensorflow","sklearn"]
for p in packages:
    try:
        m = __import__(p)
        v = getattr(m, "__version__", "N/A")
        print(f"  OK {p}: {v}")
    except Exception:
        print(f"  ERRO {p}: não instalado")
PY
echo
echo "GPU:"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -1 | \
  awk -F, "{printf \"  GPU: %s\n  Memória: %sMB/%sMB\n\", \$1,\$3,\$2}"
else
  echo "  NVIDIA GPU não detectada"
fi
echo
echo "ARQUIVOS PRINCIPAIS:"
echo "  $(ls -1 *.py *.sh *.csv *.h5 *.joblib 2>/dev/null | wc -l) arquivos no projeto"
' '''
                ui.button('9) Informações do sistema',
                          on_click=lambda: run_async(SYS_CMD)).classes('hud-btn')

        # Console
        with ui.card().classes('hud-card hud-glass grow'):
            ui.label('Console').classes('hud-title')
            global log
            log = ui.log(max_lines=400).classes('hud-log')

ui.run(title='Alzheimer HUD', host='0.0.0.0', port=8080, reload=False)