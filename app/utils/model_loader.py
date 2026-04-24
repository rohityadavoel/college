# import os
# import threading
# import numpy as np
# from PIL import Image
# import tensorflow as tf

# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess_input
# from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnetv2_preprocess_input


# class _LazyModel:
#     def __init__(self, model_path):
#         self.model_path = model_path
#         self.model = None
#         self.lock = threading.Lock()

#     def load(self):
#         if self.model is None:
#             with self.lock:
#                 if self.model is None:
#                     os.environ["CUDA_VISIBLE_DEVICES"] = ""
#                     self.model = load_model(self.model_path)
#         return self.model


# def load_keras_model(path):
#     return _LazyModel(path)


# def preprocess_mobilenetv3(img, size):
#     img = img.resize((size, size))
#     arr = np.array(img).astype(np.float32)
#     arr = np.expand_dims(arr, 0)
#     arr = mobilenet_preprocess_input(arr)
#     return arr


# def predict_binary(model_lazy, img, size):
#     model = model_lazy.load()
#     x = preprocess_mobilenetv3(img, size)
#     prob = float(model.predict(x)[0][0])
#     label = "skin" if prob >= 0.5 else "not_skin"
#     return label, prob


# def preprocess_effnetv2(img, size):
#     img = img.resize((size, size))
#     arr = np.array(img, dtype=np.float32)
#     arr = effnetv2_preprocess_input(arr)
#     arr = np.expand_dims(arr, 0)
#     return arr


# def predict_stage2(model_lazy, img, size):
#     model = model_lazy.load()
#     x = preprocess_effnetv2(img, size)
#     preds = model.predict(x)[0]
#     idx = int(np.argmax(preds))
#     prob = float(preds[idx])
#     return idx, prob




# utils/model_loader.py
import os
import threading
import zipfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnetv2_preprocess_input
import multiprocessing
import logging

log = logging.getLogger("rich")

# Tune TF threading (CPU only)
CPU_COUNT = max(1, multiprocessing.cpu_count())
tf.config.threading.set_intra_op_parallelism_threads(max(1, CPU_COUNT))
tf.config.threading.set_inter_op_parallelism_threads(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU

# -----------------------------------------
# Lazy model loader (thread-safe)
# -----------------------------------------
class _LazyModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    path = self.model_path
                    if path.endswith('.keras'):
                        stable_dir = path.replace('.keras', '_stable')
                        if not os.path.exists(stable_dir):
                            log.info(f"[yellow]⟳ Safe-extracting model to: {os.path.basename(stable_dir)}[/yellow]")
                            for attempt in range(3):
                                try:
                                    if os.path.exists(stable_dir): shutil.rmtree(stable_dir, ignore_errors=True)
                                    os.makedirs(stable_dir, exist_ok=True)
                                    with zipfile.ZipFile(path, 'r') as zip_ref:
                                        zip_ref.extractall(stable_dir)
                                    break # success
                                except Exception as e:
                                    if attempt == 2:
                                        log.error(f"[red]Failed to extract model after 3 attempts: {e}[/red]")
                                    else:
                                        import time
                                        time.sleep(2) # wait for locks to release
                        path = stable_dir

                    # load model (CPU), disable compile to avoid file locks on Windows
                    self._model = load_model(path, compile=False)
        return self._model

def load_keras_model(path):
    return _LazyModel(path)

# -----------------------------------------
# Preprocessing using OpenCV (expects RGB numpy HxWx3 dtype=uint8/float32)
# -----------------------------------------
def _to_rgb_if_bgr(img_np):
    # OpenCV reads as BGR; if array seems BGR convert to RGB
    # Heuristic: if image has more blue-ish values? We'll assume input from cv2 is BGR.
    return cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

def preprocess_mobilenetv3_from_np(img_np, size):
    """
    img_np: HxWx3 numpy array (BGR or RGB)
    returns: batched preprocessed array for MobileNetV3 ([-1,1])
    """
    # ensure color is RGB
    if img_np.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_np

    resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = mobilenet_preprocess_input(arr)  # handles [-1,1] scaling
    return arr

def preprocess_effnetv2_from_np(img_np, size):
    """
    For EfficientNetV2 preprocess_input (matches your Streamlit logic)
    """
    if img_np.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_np

    resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32)
    arr = effnetv2_preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

# -----------------------------------------
# Predict wrappers (use thread pool for predict)
# -----------------------------------------
def _predict_async(model, x):
    """
    Submit model.predict to the thread pool and return future.
    """
    return _predict_executor.submit(lambda: model.predict(x))

def predict_binary(lazy_model, img_np, size=224):
    """
    img_np: HxWx3 numpy array (BGR as cv2 returns)
    returns: label, probability
    """
    model = lazy_model.load()
    x = preprocess_mobilenetv3_from_np(img_np, size)
    # Direct predict - caller should use their own executor (e.g. asyncio loop.run_in_executor)
    preds = model.predict(x, verbose=0)
    # assuming sigmoid output shape (1,)
    prob = float(preds[0][0]) if preds.ndim == 2 or preds.shape[-1] == 1 else float(preds[0].max())
    label = "skin" if prob >= 0.5 else "not_skin"
    return label, prob

def predict_stage2(lazy_model, img_np, size=300):
    """
    EfficientNetV2-M single-model 7-class prediction.
    returns: index, probability
    """
    model = lazy_model.load()
    x = preprocess_effnetv2_from_np(img_np, size)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    prob = float(preds[idx])
    return idx, prob
