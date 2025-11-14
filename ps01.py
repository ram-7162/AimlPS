
!pip install -q librosa soundfile


import os, random, time, warnings, shutil, zipfile
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from google.colab import drive


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__, "NumPy:", np.__version__)


ZIP_FILE_DRIVE_PATH = "/content/drive/MyDrive/Colab/the-frequency-quest.zip"   
REAL_TRAIN_DIR = "/content/drive/MyDrive/Colab/the-frequency-quest/train/train"
TEST_DIR       = "/content/drive/MyDrive/Colab/the-frequency-quest/test"
SAMPLE_SUB     = "/content/drive/MyDrive/Colab/the-frequency-quest/sample_submission.csv"  


print("Mounting Google Drive...")
drive.mount('/content/drive', force_remount=True)
print("Drive mounted.")


if not os.path.isdir(REAL_TRAIN_DIR):
    print("train/train folder not found at expected path. Attempting to extract zip into VM...")
    VM_EXTRACT_DIR = "/content/audio_data_extracted"
    os.makedirs(VM_EXTRACT_DIR, exist_ok=True)
    if not os.path.isfile(ZIP_FILE_DRIVE_PATH):
        raise SystemExit(f"Zip not found at {ZIP_FILE_DRIVE_PATH}. Upload zip or adjust REAL_TRAIN_DIR.")
    shutil.copy(ZIP_FILE_DRIVE_PATH, "/content/the-frequency-quest.zip")
    with zipfile.ZipFile("/content/the-frequency-quest.zip","r") as z:
        z.extractall(VM_EXTRACT_DIR)
   
    candidates = []
    for root, dirs, files in os.walk(VM_EXTRACT_DIR):
        for d in dirs:
            if d == "train":
                try_path = os.path.join(root, d, "train")  
                if os.path.isdir(try_path):
                    candidates.append(try_path)
                elif os.path.isdir(os.path.join(root, d)):
                    candidates.append(os.path.join(root, d))
    if candidates:
        chosen = None
        for c in candidates:
            subdirs = [x for x in os.listdir(c) if os.path.isdir(os.path.join(c, x))]
            if subdirs:
                chosen = c
                break
        if chosen is None:
            chosen = candidates[0]
        REAL_TRAIN_DIR = chosen
        TEST_DIR = os.path.join(os.path.dirname(REAL_TRAIN_DIR), "test") if os.path.isdir(os.path.join(os.path.dirname(REAL_TRAIN_DIR), "test")) else TEST_DIR
        SAMPLE_SUB = os.path.join(os.path.dirname(REAL_TRAIN_DIR), "sample_submission.csv") if os.path.isfile(os.path.join(os.path.dirname(REAL_TRAIN_DIR), "sample_submission.csv")) else SAMPLE_SUB
        print("Using extracted path:", REAL_TRAIN_DIR)
    else:
        raise SystemExit("Couldn't locate train folders after extraction. Inspect archive.")


print("Final TRAIN path:", REAL_TRAIN_DIR)
print("Final TEST path:", TEST_DIR)
print("Sample submission:", SAMPLE_SUB if os.path.isfile(SAMPLE_SUB) else "(not found)")

SR = 22050            
DURATION = 4.0         
SAMPLES = int(SR * DURATION)
N_MELS = 128
HOP_LENGTH = 512
FMAX = 8000
EXPECTED_FRAMES = int(np.ceil(SAMPLES / HOP_LENGTH))
print("Samples per clip:", SAMPLES, "Expected frames:", EXPECTED_FRAMES)

def augment_waveform(y):
    shift = np.random.randint(-int(0.1*SR), int(0.1*SR))
    y = np.roll(y, shift)
    y = y + 0.001 * np.random.randn(len(y))
    if np.random.rand() < 0.3:
        steps = np.random.uniform(-1.0, 1.0)
        try:
            y = librosa.effects.pitch_shift(y, SR, steps)
        except Exception:
            pass
    return y

def waveform_to_log_mel(y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH, fmax=FMAX):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, fmax=fmax, power=2.0)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    if np.isfinite(log_mel).all():
        norm = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-9)
    else:
        norm = np.zeros_like(log_mel, dtype=np.float32)
    return norm.astype(np.float32)


print("Scanning class folders and extracting spectrograms (this may take several minutes)...")
classes = sorted([d for d in os.listdir(REAL_TRAIN_DIR) if os.path.isdir(os.path.join(REAL_TRAIN_DIR, d))])
if not classes:
    raise SystemExit(f"No class folders found under {REAL_TRAIN_DIR}")

X_list = []
y_list = []
for cls in classes:
    cls_dir = os.path.join(REAL_TRAIN_DIR, cls)
    for root, _, files in os.walk(cls_dir):
        for fname in files:
            if not fname.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                continue
            path = os.path.join(root, fname)
            try:
                y_wave, sr = librosa.load(path, sr=SR, mono=True)
               
                if len(y_wave) >= SAMPLES:
                    start = max(0, (len(y_wave) - SAMPLES)//2)
                    y_clip = y_wave[start:start+SAMPLES]
                else:
                    y_clip = np.pad(y_wave, (0, SAMPLES - len(y_wave)), mode='constant')

                spec = waveform_to_log_mel(y_clip)
                if spec.shape[1] < EXPECTED_FRAMES:
                    spec = np.pad(spec, ((0,0),(0, EXPECTED_FRAMES - spec.shape[1])), mode='constant')
                elif spec.shape[1] > EXPECTED_FRAMES:
                    spec = spec[:, :EXPECTED_FRAMES]
                X_list.append(spec[..., np.newaxis])
                y_list.append(cls)

                y_aug = augment_waveform(y_clip)
                spec_aug = waveform_to_log_mel(y_aug)
                if spec_aug.shape[1] < EXPECTED_FRAMES:
                    spec_aug = np.pad(spec_aug, ((0,0),(0, EXPECTED_FRAMES - spec_aug.shape[1])), mode='constant')
                elif spec_aug.shape[1] > EXPECTED_FRAMES:
                    spec_aug = spec_aug[:, :EXPECTED_FRAMES]
                X_list.append(spec_aug[..., np.newaxis])
                y_list.append(cls)
            except Exception as e:
                print("skip file:", path, e)
    print(f"Processed class: {cls} (collected so far: {len(X_list)})")

X = np.array(X_list, dtype=np.float32)
y_names = np.array(y_list)
print("Built dataset shapes -> X:", X.shape, "y:", y_names.shape)

le = LabelEncoder()
y = le.fit_transform(y_names)
num_classes = len(le.classes_)
print("Classes:", le.classes_, "num_classes:", num_classes)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print("Train shapes:", X_train.shape, y_train.shape, "Val shapes:", X_val.shape, y_val.shape)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print("Class weights:", class_weight_dict)

def conv_block(x, filters):
    x = layers.Conv2D(filters, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Dropout(0.2)(x)
    return x

input_shape = X_train.shape[1:]  
print("Model input shape:", input_shape)

inp = layers.Input(shape=input_shape)
x = conv_block(inp, 32)
x = conv_block(x, 64)
x = conv_block(x, 128)
x = conv_block(x, 256)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inp, out)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def make_ds(X_arr, y_arr, training=False):
    ds = tf.data.Dataset.from_tensor_slices((X_arr, y_arr))
    if training:
        ds = ds.shuffle(2048, seed=SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = make_ds(X_train, y_train, training=True)
val_ds   = make_ds(X_val, y_val, training=False)

checkpoint = callbacks.ModelCheckpoint('best_mel_cnn.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)

EPOCHS = 60
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                    callbacks=[checkpoint, reduce_lr, early_stop],
                    class_weight=class_weight_dict)


val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print("Validation accuracy:", val_acc)

print("Preparing test set features...")
test_paths = []
for root, _, files in os.walk(TEST_DIR):
    for f in files:
        if f.lower().endswith(('.wav','.mp3','.flac','.ogg')):
            test_paths.append(os.path.join(root, f))
test_paths = sorted(test_paths)
print("Found test files:", len(test_paths))

test_specs = []
test_names = []
for p in test_paths:
    try:
        y_wave, sr = librosa.load(p, sr=SR, mono=True)
        if len(y_wave) >= SAMPLES:
            start = max(0, (len(y_wave) - SAMPLES)//2)
            y_clip = y_wave[start:start+SAMPLES]
        else:
            y_clip = np.pad(y_wave, (0, SAMPLES - len(y_wave)), mode='constant')
        spec = waveform_to_log_mel(y_clip)
        if spec.shape[1] < EXPECTED_FRAMES:
            spec = np.pad(spec, ((0,0),(0, EXPECTED_FRAMES - spec.shape[1])), mode='constant')
        elif spec.shape[1] > EXPECTED_FRAMES:
            spec = spec[:, :EXPECTED_FRAMES]
        test_specs.append(spec[..., np.newaxis])
        test_names.append(os.path.basename(p))
    except Exception as e:
        print("skip test file:", p, e)

if len(test_specs) == 0:
    print("No test audio found -> skipping submission creation.")
else:
    X_test = np.array(test_specs, dtype=np.float32)
    preds_prob = model.predict(X_test, batch_size=32)
    preds_idx = np.argmax(preds_prob, axis=1)
    preds_labels = le.inverse_transform(preds_idx)

    if os.path.isfile(SAMPLE_SUB):
        sample_df = pd.read_csv(SAMPLE_SUB)
        pred_map = dict(zip(test_names, preds_labels))
        id_col = None
        for c in ['ID','id','file','filename','audio','path']:
            if c in sample_df.columns:
                id_col = c; break
        if id_col:
            sample_df['label'] = sample_df[id_col].apply(lambda v: pred_map.get(os.path.basename(str(v)),""))
            out = 'submission_mel_cnn.csv'
            sample_df.to_csv(out, index=False)
        else:
            out = 'submission_mel_cnn.csv'
            pd.DataFrame({'ID': test_names, 'label': preds_labels}).to_csv(out, index=False)
    else:
        out = 'submission_mel_cnn.csv'
        pd.DataFrame({'ID': test_names, 'label': preds_labels}).to_csv(out, index=False)

    print("Submission saved as:", out)

print("Pipeline finished.")