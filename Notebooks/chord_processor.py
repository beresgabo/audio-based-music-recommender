# chord_processor.py
# Ez a fájl CSAK a worker függvényeket tartalmazza.
# Ide NEM kerül semmi egyéb logika — csak amit a spawned processeknek tudni kell.

import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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
    except Exception:
        return index, np.zeros(576, dtype=np.float32)