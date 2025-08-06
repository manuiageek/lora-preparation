from __future__ import annotations

import argparse
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import imagehash
from PIL import Image

# --- Configuration globale ----------------------------------------------------
HASH_SIZE = 16                # Précision du pHash
SIZE_WINDOW = 1024 * 50       # ±50 Ko
THRESHOLD = 1                 # Distance max pour être considéré identique
CACHE_FILE = Path("image_hashes_cache.pkl")  # Cache persistant sur disque
LOG_FORMAT = "%(levelname)s: %(message)s"

# Tous les cœurs logiques disponibles
auto_workers = os.cpu_count() or 1


# --- Utilitaires de hash ------------------------------------------------------
def compute_hash(path: Path) -> Tuple[Path, imagehash.ImageHash] | None:
    """Retourne (chemin, pHash) ou None si erreur."""
    try:
        return path, imagehash.average_hash(Image.open(path), hash_size=HASH_SIZE)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Erreur sur %s : %s", path, exc)
        return None


# --- Cache persistant ---------------------------------------------------------
class HashCache:
    """Cache {Path: pHash} stocké en pickle pour économiser la RAM entre exécutions."""

    def __init__(self, file: Path = CACHE_FILE) -> None:
        self.file = file
        self.data: Dict[Path, imagehash.ImageHash] = self._load()

    # Chargement du pickle ------------------------------------------------------
    def _load(self) -> Dict[Path, imagehash.ImageHash]:
        if self.file.exists():
            try:
                with self.file.open("rb") as fh:
                    return pickle.load(fh)
            except Exception:
                logging.warning("Impossible de charger le cache, il sera ignoré.")
        return {}

    # Sauvegarde du pickle ------------------------------------------------------
    def save(self) -> None:
        with self.file.open("wb") as fh:
            pickle.dump(self.data, fh)

    # Accès ---------------------------------------------------------------------
    def get_or_compute(self, path: Path) -> imagehash.ImageHash | None:
        if path not in self.data:
            res = compute_hash(path)
            if res:
                _, h = res
                self.data[path] = h
        return self.data.get(path)


# --- Détection des doublons ---------------------------------------------------
def detect_duplicates(images: List[Path], cache: HashCache) -> List[Tuple[Path, Path]]:
    """Renvoie la liste des tuples (doublon, original)."""
    buckets: Dict[int, List[Path]] = {}
    for img in images:
        buckets.setdefault(int(img.stat().st_size // SIZE_WINDOW), []).append(img)

    duplicates: List[Tuple[Path, Path]] = []

    for bucket in buckets.values():
        # pHash en parallèle
        with ProcessPoolExecutor(max_workers=auto_workers) as pool:
            hashes = pool.map(cache.get_or_compute, bucket)

        path_hash: Dict[Path, imagehash.ImageHash] = {
            p: h for p, h in zip(bucket, hashes) if h
        }

        # Comparaison intra-bucket
        for img_a, hash_a in path_hash.items():
            for img_b, hash_b in path_hash.items():
                if img_a >= img_b:
                    continue
                if (hash_a - hash_b) < THRESHOLD:
                    duplicates.append((img_b, img_a))
                    break

    return duplicates


def remove_duplicates(dups: List[Tuple[Path, Path]]) -> None:
    for dup, original in dups:
        try:
            dup.unlink()
            logging.info("Supprimé %s (doublon de %s)", dup.name, original.name)
        except Exception as exc:  # noqa: BLE001
            logging.error("Échec de suppression %s : %s", dup, exc)


# --- CLI ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suppression d’images en doublon")
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(r"T:\_SELECT\TODO\Kanpekiseijo\01"),
        help="Répertoire racine à analyser",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    images = [
        p for p in Path(args.directory).rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    ]
    cache = HashCache()

    logging.info("Analyse de %d images…", len(images))
    duplicates = detect_duplicates(images, cache)
    remove_duplicates(duplicates)

    cache.save()  # persistance du travail accompli
    logging.info("Terminé à %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()