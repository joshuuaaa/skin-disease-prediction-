import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import csv
import random
import json
import gc

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 8 # Keep small
EPOCHS = 10
LEARNING_RATE = 0.0001
DATASET_CSV_PATH = r"c:\Users\joshu\OneDrive\Documents\skin disese\dataset\balanced_dataset\balanced_dataset\balanced_dataset.csv"
DATASET_ROOT_DIR = r"c:\Users\joshu\OneDrive\Documents\skin disese\dataset\balanced_dataset\balanced_dataset"
MODEL_SAVE_PATH = r"c:\Users\joshu\OneDrive\Documents\skin disese\models\skin_disease_model.h5"

def parse_path(image_path, label):
    # Load image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize
    return img, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label

def main():
    print("Loading dataset via csv module (memory efficient)...")
    
    samples = []
    classes_set = set()
    
    try:
        with open(DATASET_CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Fix path
                raw_path = row['image_path']
                category = row['category']
                
                # Robust path construction
                filename = os.path.basename(raw_path)
                category = row['category']
                
                # Construct path: Root / Category / Filename
                # This assumes the folder structure matches the category name, which we verified.
                full_path = os.path.join(DATASET_ROOT_DIR, category, filename)
                
                if os.path.exists(full_path):
                    samples.append((full_path, category))
                    classes_set.add(category)
                # else: 
                #     # Optional: print first few missing to debug
                #     pass
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Total samples found in CSV: {len(samples)}")
    
    if not samples:
        print("No samples found.")
        return

    # Verify first file exists
    if not os.path.exists(samples[0][0]):
        print(f"Sample path check failed: {samples[0][0]}")
        # Proceeding anyway? No, if root is wrong, all wrong.
        # But we are in "try hard" mode.
    
    # Sort classes for consistency
    classes = sorted(list(classes_set))
    class_to_index = {name: i for i, name in enumerate(classes)}
    print(f"Unique classes: {len(classes)}")

    # Shuffle samples
    random.shuffle(samples)

    # Manual Split 80/20
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    print(f"Training: {len(train_samples)}, Validation: {len(val_samples)}")
    
    # Free memory of full list?
    del samples
    gc.collect()

    # Prepare for tf.data
    # We need separate lists for paths and labels
    # Generator approach to avoid creating huge lists? 
    # Or just lists. Lists of 200k tuples vs lists of 200k strings + 200k ints.
    # We can use a generator for tf.data.Dataset.from_generator! This is MOST memory efficient.
    
    def generator(sample_list):
        for path, cat in sample_list:
            yield path, class_to_index[cat]

    # Need simpler generator functions that don't take args for `from_generator` signature (or use partial)
    # Actually, simpler: create lists. It's faster.
    # List of 200k strings is ok.
    
    train_paths = [s[0] for s in train_samples]
    train_labels = [class_to_index[s[1]] for s in train_samples]
    
    val_paths = [s[0] for s in val_samples]
    val_labels = [class_to_index[s[1]] for s in val_samples]
    
    # One-hot encoding happening inside model? No, using sparse_categorical_crossentropy is better for memory!
    # If we use sparse_categorical_crossentropy, labels are just ints.
    # If we use categorical_crossentropy, we need one-hot vectors (floats).
    # Using sparse is much better for memory.
    
    # Update Model compile loss to sparse_categorical_crossentropy
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.map(parse_path, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=100)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(parse_path, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(classes), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    # CHANGED TO SPARSE
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    
    print("Starting training...")
    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save indices
    inverted_indices = {str(v): k for k, v in class_to_index.items()}
    with open(os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'class_indices.json'), 'w') as f:
        json.dump(inverted_indices, f)
    print("Saved class indices.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")
