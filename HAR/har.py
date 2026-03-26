import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import pandas as pd
import glob, os
import struct
import warnings
from pathlib import Path
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin

class Standard_Scaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X


HAR1_NUM_PACKETS = 150
HAR1_NFFT = 128
HAR1_HEADER_OFFSET_WORDS = 16
HAR1_REMOVE_SUBCARRIERS = np.array(
    [1, 2, 3, 4, 5, 6, 12, 40, 54, 62, 63, 64, 65, 66, 67, 68, 76, 90, 118, 124, 125, 126, 127, 128],
    dtype=np.int64,
) - 1
HAR1_CLASS_DIRS = {
    "Empty": "EMPTY",
    "Sit": "SIT",
    "Stand": "STAND",
    "Walk": "WALK",
}


def _read_har1_pcap(filepath: Path, num_packets: int = HAR1_NUM_PACKETS) -> tuple[np.ndarray, int]:
    samples = []

    with filepath.open("rb") as f:
        global_header = f.read(24)
        if len(global_header) != 24:
            raise ValueError(f"Incomplete PCAP global header: {filepath}")

        while len(samples) < num_packets:
            header = f.read(16)
            if len(header) < 16:
                break

            _, _, incl_len, orig_len = struct.unpack("<IIII", header)
            payload = f.read(incl_len)
            if len(payload) < incl_len:
                break

            if orig_len - (HAR1_HEADER_OFFSET_WORDS - 1) * 4 != HAR1_NFFT * 4:
                continue

            words = np.frombuffer(payload, dtype="<u4")
            if words.size < HAR1_HEADER_OFFSET_WORDS - 1 + HAR1_NFFT:
                continue

            csi_words = words[HAR1_HEADER_OFFSET_WORDS - 1: HAR1_HEADER_OFFSET_WORDS - 1 + HAR1_NFFT]
            iq_pairs = csi_words.view("<i2").reshape(-1, 2)
            csi = iq_pairs[:, 0].astype(np.float32) + 1j * iq_pairs[:, 1].astype(np.float32)
            csi = np.abs(np.fft.fftshift(csi))
            csi = np.delete(csi, HAR1_REMOVE_SUBCARRIERS)
            samples.append(csi.astype(np.float32, copy=False))

    if not samples:
        raise ValueError(f"{filepath} did not yield any valid CSI packets")

    valid_packets = len(samples)

    if valid_packets < num_packets:
        missing_packets = num_packets - valid_packets
        samples.extend([samples[-1].copy() for _ in range(missing_packets)])

    return np.stack(samples[:num_packets], axis=0), valid_packets


def _build_har1_from_raw(dataset_dir: Path):
    raw_root = dataset_dir / "Experiment-1" / "Experiment-1" / "Train" / "Data8_Train" / "room" / "Red"
    if not raw_root.exists():
        raise FileNotFoundError(
            "HAR Experiment-1 preprocessing fallback could not find raw PCAP files under "
            f"{raw_root}"
        )

    samples = []
    labels = []
    padded_files = []

    for label_name, class_dir in HAR1_CLASS_DIRS.items():
        class_root = raw_root / class_dir
        pcap_files = sorted(class_root.glob("*.pcap"))
        if not pcap_files:
            raise FileNotFoundError(f"HAR Experiment-1 class folder is empty: {class_root}")

        for filepath in pcap_files:
            sample, valid_packets = _read_har1_pcap(filepath)
            samples.append(sample)
            labels.append(label_name)
            if valid_packets < HAR1_NUM_PACKETS:
                padded_files.append((filepath, valid_packets))

    if padded_files:
        max_missing_packets = max(HAR1_NUM_PACKETS - valid_packets for _, valid_packets in padded_files)
        example_paths = ", ".join(
            f"{path.name} ({valid_packets}/{HAR1_NUM_PACKETS})" for path, valid_packets in padded_files[:3]
        )
        warnings.warn(
            f"HAR Experiment-1 raw preprocessing padded {len(padded_files)} capture(s) that contained fewer than "
            f"{HAR1_NUM_PACKETS} valid CSI packets. Worst deficit: {max_missing_packets} packet(s). "
            f"Examples: {example_paths}",
            RuntimeWarning,
        )

    data = np.stack(samples, axis=0)
    class_labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        class_labels,
        test_size=0.2,
        random_state=42,
        stratify=class_labels,
    )

    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.transform(y_test)

    scaler = Standard_Scaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_test = scaler.transform(X_test).astype(np.float32, copy=False)
    y_train = y_train.astype(np.float32, copy=False)
    y_test = y_test.astype(np.float32, copy=False)

    np.save(dataset_dir / "X_train.npy", X_train)
    np.save(dataset_dir / "X_test.npy", X_test)
    np.save(dataset_dir / "y_train.npy", y_train)
    np.save(dataset_dir / "y_test.npy", y_test)

    return X_train, y_train, X_test, y_test
    
def har1():
    dataset_dir = Path(__file__).resolve().parent
    x_train_path = dataset_dir / 'X_train.npy'
    x_test_path = dataset_dir / 'X_test.npy'
    y_train_path = dataset_dir / 'y_train.npy'
    y_test_path = dataset_dir / 'y_test.npy'

    if not all(path.exists() for path in [x_train_path, x_test_path, y_train_path, y_test_path]):
        X_train, y_train, X_test, y_test = _build_har1_from_raw(dataset_dir)
        print("HAR EXPERIMENT-1 train X,y shape is ",X_train.shape, y_train.shape)
        return X_train, y_train, X_test, y_test

    # Load the data
    X_train = np.load(x_train_path)
    X_test = np.load(x_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)

    if y_train.ndim == 1 or (y_train.ndim == 2 and y_train.shape[1] == 1):
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train.ravel())
        y_test_encoded = label_encoder.transform(y_test.ravel())
        label_binizer = LabelBinarizer()
        y_train = label_binizer.fit_transform(y_train_encoded)
        y_test = label_binizer.transform(y_test_encoded)

    print("HAR EXPERIMENT-1 train X,y shape is ",X_train.shape, y_train.shape)
    return X_train, y_train, X_test, y_test


# function for reading CSV files 
def reading_file(activity_csv):     
    results = []
    for i in range(len(activity_csv)):
        df = pd.read_csv(activity_csv[i])
        results.append(df.values.astype(np.float32))  
    return results
#function for labeling the samples 
def label(activity, label):
    list_y = []
    for i in range(len(activity)):
        list_y.append(label)
    return np.array(list_y).reshape(-1, 1) 

def har3():
    dataset_dir = Path(__file__).resolve().parent / 'Experiment-3' / 'Data'
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"HAR Experiment-3 data folder not found: {dataset_dir}"
        )

    list_file = [path.name for path in dataset_dir.glob("*.csv")]

    empty_csv = [str(dataset_dir / i) for i in list_file if i.startswith('Empty')] #list for empty csv files 
    lying_csv = [str(dataset_dir / i) for i in list_file if i.startswith('Lying')] #list for lying csv files 
    sitting_csv = [str(dataset_dir / i) for i in list_file if i.startswith('Sitting')] #list for sitting csv files 
    standing_csv = [str(dataset_dir / i) for i in list_file if i.startswith('Standing')] #list for satnding csv files 
    walking_csv = [str(dataset_dir / i) for i in list_file if i.startswith('Walking')] #list for walking csv files 

    #calling reading_file function  
    empty = reading_file(empty_csv) 
    lying = reading_file(lying_csv)
    sitting = reading_file(sitting_csv)
    standing = reading_file(standing_csv)
    walking = reading_file(walking_csv)

    walking_label = label(walking, 'walking') 
    empty_label = label(empty, 'empty') 
    lying_label = label(lying, 'lying') 
    sitting_label = label(sitting, 'sitting') 
    standing_label = label(standing, 'standing') 

    #concatenate all the samples into one np array 
    array_tuple = (empty, lying, sitting,standing, walking)
    data_X = np.vstack(array_tuple)

    #concatenate all the label into one array 
    label_tuple = (empty_label, lying_label, sitting_label,standing_label,  walking_label)
    data_y = np.vstack(label_tuple)

    #randomize the sample 

    data, labels = shuffle(data_X, data_y)

    label_binarizer = LabelBinarizer()
    labels = label_binarizer.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=42)

    sc = Standard_Scaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform (X_test)

    print("HAR EXPERIMENT-3 train X, y shape is ",X_train.shape, y_train.shape)
    return X_train, y_train, X_test, y_test
