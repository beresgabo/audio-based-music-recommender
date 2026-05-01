# chord_worker.py
import os
import h5py
import numpy as np
from tqdm import tqdm
import concurrent.futures
import warnings
import json

warnings.filterwarnings("ignore")

H5_PATH = "../Dataset/spotify_dataset_compressed.h5"
MP3_DIR = "../MP3/"
CHECKPOINT_PATH = "chord_checkpoint.json"
MAX_WORKERS = 5   # 5 worker * ~800MB = ~4GB RAM, biztonságos 32GB-nál
BATCH_SIZE = 200  # Ennyi dal után ment checkpointot és írja ki a H5-be

CHORD_MAP = {
    'C:maj': 0, 'C#:maj': 1, 'D:maj': 2, 'D#:maj': 3, 'E:maj': 4, 'F:maj': 5,
    'F#:maj': 6, 'G:maj': 7, 'G#:maj': 8, 'A:maj': 9, 'A#:maj': 10, 'B:maj': 11,
    'C:min': 12, 'C#:min': 13, 'D:min': 14, 'D#:min': 15, 'E:min': 16, 'F:min': 17,
    'F#:min': 18, 'G:min': 19, 'G#:min': 20, 'A:min': 21, 'A#:min': 22, 'B:min': 23,
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6,
    'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}

global_feat_processor = None
global_chord_processor = None

def init_worker():
    global global_feat_processor, global_chord_processor
    from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor
    global_feat_processor = CNNChordFeatureProcessor()
    global_chord_processor = CRFChordRecognitionProcessor()

def process_single_track(args):
    index, mp3_path = args
    if not os.path.exists(mp3_path):
        return index, np.zeros(576, dtype=np.float32)
    try:
        features = global_feat_processor(mp3_path)
        chords_output = global_chord_processor(features)
        numeric_sequence = [
            CHORD_MAP[label]
            for _, _, label in chords_output
            if label != 'N' and label in CHORD_MAP
        ]
        if not numeric_sequence:
            return index, np.zeros(576, dtype=np.float32)
        compressed_seq = [numeric_sequence[0]]
        for chord in numeric_sequence[1:]:
            if chord != compressed_seq[-1]:
                compressed_seq.append(chord)
        transition_matrix = np.zeros((24, 24), dtype=np.float32) + 1e-5
        for i in range(len(compressed_seq) - 1):
            transition_matrix[compressed_seq[i], compressed_seq[i + 1]] += 1
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        return index, (transition_matrix / row_sums).flatten()
    except Exception as e:
        return index, np.zeros(576, dtype=np.float32)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            data = json.load(f)
        print(f"   ✅ Checkpoint betöltve: {data['completed']} dal már kész, folytatás innen...")
        return set(data["done_indices"])
    return set()

def save_checkpoint(done_indices):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"completed": len(done_indices), "done_indices": list(done_indices)}, f)

def main():
    print("1. Metaadatok beolvasása...")
    with h5py.File(H5_PATH, "r") as hf:
        t_uris  = hf["tracks/track_uri"][:]
        t_names = hf["tracks/track_name"][:]
        a_names = hf["tracks/artist_name"][:]
        ml_uris = hf["ml/track_uri"][:]

    track_info_dict = {}
    for i in range(len(t_uris)):
        uri    = t_uris[i].decode('utf-8') if isinstance(t_uris[i], bytes) else t_uris[i]
        name   = t_names[i].decode('utf-8') if isinstance(t_names[i], bytes) else t_names[i]
        artist = a_names[i].decode('utf-8') if isinstance(a_names[i], bytes) else a_names[i]
        track_info_dict[uri] = (name, artist)

    total = len(ml_uris)
    print(f"   Összes dal: {total}")

    # Összes task összeállítása
    all_tasks = []
    for i in range(total):
        ml_uri = ml_uris[i].decode('utf-8') if isinstance(ml_uris[i], bytes) else ml_uris[i]
        if ml_uri in track_info_dict:
            t_name, a_name = track_info_dict[ml_uri]
            safe_name = (
                f"{t_name} - {a_name}"
                .replace('/', '-').replace('\\', '-').replace(':', '-')
                .replace('*', '-').replace('?', '-').replace('"', '-')
                .replace('<', '-').replace('>', '-').replace('|', '-')
            )
            all_tasks.append((i, os.path.join(MP3_DIR, f"{safe_name}.mp3")))
        else:
            all_tasks.append((i, "NOT_FOUND"))

    # Checkpoint: kihagyjuk a már kész dalokat
    done_indices = load_checkpoint()
    pending_tasks = [t for t in all_tasks if t[0] not in done_indices]
    print(f"   Feldolgozandó (még nem kész): {len(pending_tasks)} dal")

    if not pending_tasks:
        print("✅ Minden dal már fel van dolgozva!")
        return

    # H5 dataset méret ellenőrzés
    with h5py.File(H5_PATH, "r+") as hf:
        dset = hf["features/markov_chords"]
        if dset.shape[0] < total:
            dset.resize((total, 576))

    print(f"2. Feldolgozás indítása {MAX_WORKERS} workerrel...")

    # Batch-enkénti feldolgozás + mentés
    for batch_start in range(0, len(pending_tasks), BATCH_SIZE):
        batch = pending_tasks[batch_start : batch_start + BATCH_SIZE]
        batch_results = {}

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=MAX_WORKERS, initializer=init_worker
        ) as executor:
            futures = {executor.submit(process_single_track, task): task for task in batch}
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(batch),
                desc=f"Batch {batch_start // BATCH_SIZE + 1} / {-(-len(pending_tasks) // BATCH_SIZE)}"
            ):
                try:
                    index, result_vector = future.result()
                    batch_results[index] = result_vector
                except Exception as e:
                    task = futures[future]
                    print(f"  [HIBA] index={task[0]}: {e}")
                    batch_results[task[0]] = np.zeros(576, dtype=np.float32)

        # Batch eredmények H5-be írása
        with h5py.File(H5_PATH, "r+") as hf:
            dset = hf["features/markov_chords"]
            for index, vector in batch_results.items():
                dset[index] = vector

        # Checkpoint frissítése
        done_indices.update(batch_results.keys())
        save_checkpoint(done_indices)
        print(f"   💾 Checkpoint mentve: {len(done_indices)}/{total} kész")

    # Sikeres befejezés után checkpoint törlése
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    print("\n✅ Minden dal feldolgozva és mentve!")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()