import os, time, statistics, math, platform, subprocess
import numpy as np
import tensorflow as tf

# ---- Helper: detectar GPU (nome e memória) ----
def detect_gpu_tf():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return "CPU", None
        # Tenta via nvidia-smi (se disponível)
        try:
            name = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                encoding="utf-8"
            ).strip().splitlines()[0]
            mem_total = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            ).strip().splitlines()[0]
            return name, f"{int(mem_total)} MB"
        except Exception:
            # Fallback simples
            try:
                details = tf.config.experimental.get_device_details(tf.config.list_logical_devices('GPU')[0])
                name = details.get('device_name', 'GPU')
                return name, None
            except Exception:
                return "GPU", None
    except Exception:
        return "Desconhecido", None

# ---- Callback para cronometrar ----
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.run_start = time.perf_counter()
        self.epoch_times = []
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.perf_counter()
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.perf_counter() - self.epoch_start)
    def on_train_end(self, logs=None):
        self.total_time = time.perf_counter() - self.run_start

def pretty_minutes(seconds):
    m = seconds / 60.0
    return f"{m:.2f} min"

# ---- Exemplo de criação do modelo (substitua pelo seu) ----
def build_model(input_shape, n_classes=2):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation="softmax" if n_classes > 1 else "sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy" if n_classes > 1 else "binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ---- Dados de exemplo (substitua pelos seus X_train, y_train) ----
# X_train: numpy array shape (N, D)
# y_train: numpy array shape (N,)
N, D = 2000, 42
X_train = np.random.randn(N, D).astype("float32")
y_train = np.random.randint(0, 2, size=(N,)).astype("int32")
X_val   = np.random.randn(400, D).astype("float32")
y_val   = np.random.randint(0, 2, size=(400,)).astype("int32")

# ---- Parâmetros ----
EPOCHS = 50
BATCH_SIZE = 64
RUNS = 3  # quantas execuções você quer medir

all_times = []
gpu_name, gpu_mem = detect_gpu_tf()
print(f"Dispositivo: {gpu_name}" + (f" | Memória: {gpu_mem}" if gpu_mem else ""))

for r in range(1, RUNS + 1):
    print(f"\n>>> Execução {r}/{RUNS}")
    model = build_model(input_shape=(D,), n_classes=2)
    th = TimeHistory()
    # Callbacks usuais (adicione EarlyStopping/ReduceLROnPlateau se quiser)
    cbs = [th]
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=cbs,
    )
    print(f"Tempo total: {pretty_minutes(th.total_time)} "
          f"(média por época ~ {th.total_time/EPOCHS:.2f}s)")
    all_times.append(th.total_time)

# ---- Estatísticas finais ----
mean_sec = statistics.mean(all_times)
std_sec  = statistics.pstdev(all_times)
print("\n=== Resumo de Tempo ===")
print(f"Tempo médio por execução: {pretty_minutes(mean_sec)} "
      f"(± {std_sec/60:.2f} min) em ambiente {gpu_name}"
      + (f" ({gpu_mem})" if gpu_mem else ""))