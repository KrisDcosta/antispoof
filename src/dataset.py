"""
ASVspoof 2019 LA dataset loader.

Supports two protocol modes:
  - CM  (preferred): official countermeasure protocols with separate train/dev/eval
  - ASV (fallback):  uses ASV gi trial list, deduplicates by file_id, applies
                     an 80/20 internal split. Use until CM protocols are downloaded.

CM protocol layout (preferred):
    data/LA/ASVspoof2019_LA_cm_protocols/
        ASVspoof2019.LA.cm.train.trn.txt
        ASVspoof2019.LA.cm.dev.trl.txt
        ASVspoof2019.LA.cm.eval.trl.txt

ASV protocol layout (fallback, included in standard LA download):
    data/LA/ASVspoof2019_LA_asv_protocols/
        ASVspoof2019.LA.asv.dev.gi.trl.txt

Protocol file format (space-separated, both modes):
    SPEAKER_ID  FILE_ID  SYSTEM_ID  KEY
    where KEY is 'bonafide' or 'spoof'
"""

import os
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Sample:
    file_id: str
    path: str
    label: int        # 1 = bonafide, 0 = spoof
    system_id: str    # '-' for bonafide, A01-A19 for spoof


# --- CM protocol paths (official, preferred) ---

CM_PROTOCOL_FILES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev":   "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval":  "ASVspoof2019.LA.cm.eval.trl.txt",
}

CM_AUDIO_DIRS = {
    "train": "ASVspoof2019_LA_train",
    "dev":   "ASVspoof2019_LA_dev",
    "eval":  "ASVspoof2019_LA_eval",
}

# --- ASV protocol path (fallback) ---

ASV_DEV_GI = "ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.gi.trl.txt"
ASV_AUDIO_DIR = "ASVspoof2019_LA_dev/flac"


def _has_cm_protocols(data_root: str) -> bool:
    cm_dir = os.path.join(data_root, "ASVspoof2019_LA_cm_protocols")
    return os.path.isdir(cm_dir) and any(
        os.path.exists(os.path.join(cm_dir, f))
        for f in CM_PROTOCOL_FILES.values()
    )


def _parse_line(parts: list) -> tuple:
    """
    Parse one protocol line into (file_id, system_id, label).

    CM format  (5 fields): SPEAKER FILE_ID SYSTEM_ID _ KEY
        bonafide: LA_0069 LA_T_xxx - - bonafide
        spoof:    LA_0069 LA_T_xxx A04 A04 spoof

    ASV format (4 fields): SPEAKER FILE_ID SYSTEM_OR_KEY VERDICT
        bonafide: LA_0073 LA_D_xxx bonafide target
        spoof:    LA_0078 LA_D_xxx A01 spoof
    """
    file_id = parts[1]

    if len(parts) == 5:
        # CM protocol: SPEAKER FILE_ID - SYSTEM_ID KEY
        # parts[2] is always '-'; actual attack ID is in parts[3]
        system_id = parts[3]                          # '-' for bonafide, 'A01'-'A19' for spoof
        label     = 1 if parts[4] == "bonafide" else 0
    elif len(parts) == 4:
        # ASV protocol: SPEAKER FILE_ID SYSTEM_OR_BONAFIDE VERDICT
        # parts[2] = "bonafide" for real speech (target or nontarget speaker)
        # parts[2] = "A01"-"A19" for synthetic speech
        # Both target and nontarget trials with "bonafide" speech = label 1 for CM task
        if parts[2] == "bonafide":
            system_id = "-"
            label     = 1
        else:
            system_id = parts[2]                      # 'A01'-'A19'
            label     = 0
    else:
        return None

    return file_id, system_id, label


def _read_protocol(protocol_path: str, audio_dir: str,
                   limit: int = None) -> List[Sample]:
    """Parse a protocol file, skip missing audio, deduplicate by file_id."""
    seen = set()
    samples = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            parsed = _parse_line(parts)
            if parsed is None:
                continue
            file_id, system_id, label = parsed

            if file_id in seen:
                continue
            seen.add(file_id)

            audio_path = os.path.join(audio_dir, f"{file_id}.flac")
            if not os.path.exists(audio_path):
                continue

            samples.append(Sample(file_id=file_id, path=audio_path,
                                  label=label, system_id=system_id))
            if limit is not None and len(samples) >= limit:
                break
    return samples


def load_split(data_root: str, split: str,
               limit: int = None) -> List[Sample]:
    """
    Load samples for one split.

    Uses CM protocols if available; falls back to ASV gi protocol
    with an internal 80/20 split (train=80%, dev=20%).

    Args:
        data_root: path to data/LA/
        split:     'train', 'dev', or 'eval'
        limit:     cap on samples returned (None = all)
    """
    if _has_cm_protocols(data_root):
        return _load_cm_split(data_root, split, limit)
    else:
        if split == "eval":
            raise ValueError(
                "eval split requires CM protocols. "
                "Download ASVspoof2019_LA_cm_protocols and place under data/LA/"
            )
        print(f"  [dataset] CM protocols not found — using ASV gi fallback "
              f"(80/20 split). Download CM protocols for official evaluation.")
        return _load_asv_fallback(data_root, split, limit)


def _load_cm_split(data_root: str, split: str, limit: int) -> List[Sample]:
    if split not in CM_PROTOCOL_FILES:
        raise ValueError(f"split must be one of {list(CM_PROTOCOL_FILES)}, got '{split}'")
    protocol_path = os.path.join(data_root, "ASVspoof2019_LA_cm_protocols",
                                 CM_PROTOCOL_FILES[split])
    audio_dir = os.path.join(data_root, CM_AUDIO_DIRS[split], "flac")

    if not os.path.exists(protocol_path):
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    return _read_protocol(protocol_path, audio_dir, limit)


def _load_asv_fallback(data_root: str, split: str,
                       limit: int) -> Tuple[List[Sample], List[Sample]]:
    protocol_path = os.path.join(data_root, ASV_DEV_GI)
    audio_dir     = os.path.join(data_root, ASV_AUDIO_DIR)

    if not os.path.exists(protocol_path):
        raise FileNotFoundError(
            f"ASV gi protocol not found: {protocol_path}\n"
            "Ensure the LA dataset is placed under data/LA/"
        )

    all_samples = _read_protocol(protocol_path, audio_dir, limit=None)

    # Deterministic 80/20 split
    random.seed(42)
    shuffled = all_samples[:]
    random.shuffle(shuffled)
    cut = int(len(shuffled) * 0.8)
    train_samples = shuffled[:cut]
    dev_samples   = shuffled[cut:]

    if limit is not None:
        train_samples = train_samples[:limit]
        dev_samples   = dev_samples[:limit]

    return train_samples if split == "train" else dev_samples


def split_summary(samples: List[Sample]) -> dict:
    bonafide = sum(1 for s in samples if s.label == 1)
    spoof    = len(samples) - bonafide
    systems  = sorted(set(s.system_id for s in samples if s.system_id != "-"))
    return {"total": len(samples), "bonafide": bonafide,
            "spoof": spoof, "systems": systems}
