"""
cnn_malaria_submit_fixed.py
Complete pipeline for Malaria detection using the NIH Malaria Cell Images dataset.

Place this file in the same folder that contains `cell_images/` (which has Parasitized/ and Uninfected/).
"""

import os
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report

# Reduce verbose TF logs (INFO) BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -----------------------------
# Config
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BATCH_SIZE = 32
IMG_SIZE = (64, 64)
EPOCHS = 10  # adjust if you want more/less
BASE_PATH = os.path.dirname(os.path.abspath(__file__))  # script folder
DATA_DIR = os.path.join(BASE_PATH, "cell_images")  # <- your dataset folder
SPLIT_DIR = os.path.join(DATA_DIR, "split_data")   # where train/test will be created
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "malaria_cnn.h5")
HISTORY_PNG = os.path.join(BASE_PATH, "training_history.png")
CM_PNG = os.path.join(BASE_PATH, "confusion_matrix.png")

# -----------------------------
# 1) Check dataset presence
# -----------------------------
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}\nMake sure 'cell_images' is in the same folder as this script.")

# expected original structure: cell_images/Parasitized & cell_images/Uninfected
for cls in ("Parasitized", "Uninfected"):
    if not os.path.exists(os.path.join(DATA_DIR, cls)):
        raise FileNotFoundError(f"Expected folder missing: {os.path.join(DATA_DIR, cls)}")

# -----------------------------
# 2) Create split_data (80/20) if not already present
# -----------------------------
def create_split_data(src_root, target_root, split_ratio=0.8):
    # If split folder exists and has files, skip splitting
    if os.path.exists(target_root):
        # check if it contains images under train/test
        for split in ("train", "test"):
            for category in ("Parasitized", "Uninfected"):
                p = os.path.join(target_root, split, category)
                if os.path.exists(p) and any(os.scandir(p)):
                    print("Split data folder already exists and contains data, skipping split.")
                    return

    os.makedirs(target_root, exist_ok=True)
    for split in ("train", "test"):
        for category in ("Parasitized", "Uninfected"):
            os.makedirs(os.path.join(target_root, split, category), exist_ok=True)

    print("Splitting dataset into train/test (80/20)...")
    for category in ("Parasitized", "Uninfected"):
        src_folder = os.path.join(src_root, category)
        files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
        random.shuffle(files)
        split_idx = int(len(files) * split_ratio)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        for f in train_files:
            shutil.copy(os.path.join(src_folder, f), os.path.join(target_root, "train", category, f))
        for f in test_files:
            shutil.copy(os.path.join(src_folder, f), os.path.join(target_root, "test", category, f))

    print("Dataset split complete.")
    print(f"Train: {os.path.join(target_root, 'train')}")
    print(f"Test:  {os.path.join(target_root, 'test')}")

create_split_data(DATA_DIR, SPLIT_DIR, split_ratio=0.8)

train_dir = os.path.join(SPLIT_DIR, "train")
test_dir  = os.path.join(SPLIT_DIR, "test")

# -----------------------------
# 3) Data generators
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    seed=SEED
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # important for confusion matrix / predictions
)

# -----------------------------
# 4) Build the CNN model
# -----------------------------
def build_model(input_shape=(64,64,3)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# 5) Callbacks
# -----------------------------
callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1)
]

# -----------------------------
# 6) Train
# -----------------------------
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks
)

# Save final model (best saved by checkpoint already)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# -----------------------------
# 7) Evaluate & plots
# -----------------------------
# Evaluate on test set
loss, acc = model.evaluate(test_generator, verbose=1)
print(f"Test loss: {loss:.4f} - Test accuracy: {acc:.4f}")

# Plot training history
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history.get('loss', []), label='train_loss')
plt.plot(history.history.get('val_loss', []), label='val_loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history.get('accuracy', []), label='train_acc')
plt.plot(history.history.get('val_accuracy', []), label='val_acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig(HISTORY_PNG)
print(f"Training history saved to {HISTORY_PNG}")
plt.show()

# -----------------------------
# 8) Confusion matrix & classification report
# -----------------------------
# Get predictions (probabilities) and convert to binary labels
test_generator.reset()
pred_probs = model.predict(test_generator, verbose=1)
pred_labels = (pred_probs.ravel() > 0.5).astype(int)

true_labels = test_generator.classes  # numeric labels according to flow_from_directory

cm = confusion_matrix(true_labels, pred_labels)
print("Confusion matrix:\n", cm)
print("\nClassification report:\n", classification_report(true_labels, pred_labels, target_names=list(test_generator.class_indices.keys())))

# Plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues):
    plt.figure(figsize=(6,5))
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_to_plot = cm_norm
        print("Normalized confusion matrix")
    else:
        cm_to_plot = cm
        print('Confusion matrix, without normalization')

    plt.imshow(cm_to_plot, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm_to_plot.max() / 2.
    for i, j in itertools.product(range(cm_to_plot.shape[0]), range(cm_to_plot.shape[1])):
        plt.text(j, i, format(cm_to_plot[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_to_plot[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(CM_PNG)
    print(f"Confusion matrix image saved to {CM_PNG}")
    plt.show()

class_names = list(test_generator.class_indices.keys())
plot_confusion_matrix(cm, class_names, normalize=False)

# -----------------------------
# 9) Predict random images from test set (one per class) and display
# -----------------------------
from tensorflow.keras.preprocessing import image

def predict_and_show(img_path, ax, true_label):
    # Load and preprocess
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_arr = image.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_arr, axis=0)

    # Predict
    prob = model.predict(img_batch, verbose=0)[0][0]
    label = "Parasitized" if prob > 0.5 else "Uninfected"

    # Plot on given ax
    ax.imshow(img_arr.astype("float32"))
    ax.set_title(f"True: {true_label}\nPred: {label} ({prob:.2f})", fontsize=9)
    ax.axis("off")
    return label, prob

# prepare figure
fig, axes = plt.subplots(1, 2, figsize=(8,4))
samples_info = []

for idx, category in enumerate(["Parasitized", "Uninfected"]):
    test_cat_dir = os.path.join(test_dir, category)
    files = [f for f in os.listdir(test_cat_dir) if os.path.isfile(os.path.join(test_cat_dir, f))]
    if files:
        chosen = random.choice(files)
        sample_path = os.path.join(test_cat_dir, chosen)
        print(f"\n✅ Using sample image: {sample_path}")
        lbl, pr = predict_and_show(sample_path, axes[idx], category)
        samples_info.append((sample_path, lbl, pr))
    else:
        axes[idx].text(0.5, 0.5, "No images", horizontalalignment='center')
        axes[idx].axis('off')

plt.tight_layout()
plt.show()

print("\nSummary of sample predictions:")
for sp, lbl, pr in samples_info:
    print(f"{sp} -> Predicted: {lbl}, Probability (Parasitized): {pr:.4f}")
