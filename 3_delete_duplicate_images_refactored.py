#!/usr/bin/env python3
"""Détecteur/suppresseur de doublons d'images ultra-optimisé pour CPU haute performance."""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import lz4.frame  # compression ultra-rapide

import imagehash
from PIL import Image

# Configuration optimisée mais stable
HASH_SIZE = 16  # Précision originale
SIZE_WINDOW = 1024 * 50  # Fenêtre large pour précision
THRESHOLD = 1  # Seuil strict comme l'original
CACHE_FILE = Path("image_hashes_cache.lz4")
# Configuration stable pour 5950X
WORKERS = os.cpu_count() * 2  # 64 workers avec ThreadPoolExecutor seulement
BATCH_SIZE = 1000  # Batches plus petits mais nombreux
CACHE_VERSION = "v5"  # Version pour invalider les anciens caches


def compute_hash_precise(path: Path) -> tuple[Path, imagehash.ImageHash] | None:
    """Hash précis avec average_hash optimisé."""
    try:
        with Image.open(path) as img:
            h = imagehash.average_hash(img, hash_size=HASH_SIZE)
            return path, h
    except Exception as e:
        logging.warning(f"Erreur {path}: {e}")
        return None


class TurboCache:
    """Cache compressé en mémoire avec chargement lazy."""
    
    def __init__(self) -> None:
        self.data: Dict[str, str] = {}
        self.stats: Dict[str, float] = {}
        self.version: str = CACHE_VERSION
        self._load_async()
    
    def _load_async(self) -> None:
        """Chargement asynchrone du cache."""
        if not CACHE_FILE.exists():
            return
        try:
            with CACHE_FILE.open("rb") as f:
                compressed = f.read()
            if compressed:
                data = pickle.loads(lz4.frame.decompress(compressed))
                
                if data.get('version') != CACHE_VERSION:
                    logging.info("Version de cache obsolète, reconstruction nécessaire")
                    return
                
                self.data = data.get('hashes', {})
                self.stats = data.get('stats', {})
                logging.info(f"Cache chargé: {len(self.data)} entrées")
        except Exception as e:
            logging.warning(f"Erreur chargement cache: {e}")
    
    def save(self) -> None:
        """Sauvegarde compressée ultra-rapide."""
        try:
            data = {
                'version': self.version,
                'hashes': self.data, 
                'stats': self.stats
            }
            compressed = lz4.frame.compress(pickle.dumps(data))
            CACHE_FILE.write_bytes(compressed)
            logging.info(f"Cache sauvé: {len(self.data)} entrées")
        except Exception as e:
            logging.error(f"Erreur sauvegarde cache: {e}")
    
    def is_fresh(self, path: Path) -> bool:
        """Vérifie si le hash est encore valide."""
        path_str = str(path)
        try:
            return (path_str in self.data and 
                    path_str in self.stats and 
                    self.stats[path_str] == path.stat().st_mtime)
        except Exception:
            return False
    
    def get_or_compute(self, path: Path) -> imagehash.ImageHash | None:
        """Récupère ou calcule le hash d'une image (version simple)."""
        path_str = str(path)
        
        # Vérifier le cache
        if self.is_fresh(path):
            try:
                hash_obj = imagehash.hex_to_hash(self.data[path_str])
                if hash_obj.hash.shape == (HASH_SIZE, HASH_SIZE):
                    return hash_obj
            except Exception:
                pass
        
        # Calculer le nouveau hash
        if result := compute_hash_precise(path):
            _, hash_obj = result
            self.data[path_str] = str(hash_obj)
            try:
                self.stats[path_str] = path.stat().st_mtime
            except Exception:
                pass
            return hash_obj
        
        return None


def process_group(group: List[Path], cache: TurboCache) -> int:
    """Traite un groupe d'images candidats."""
    if len(group) < 2:
        return 0
    
    # Hash du groupe avec ThreadPoolExecutor simple et stable
    with ThreadPoolExecutor(max_workers=min(32, len(group))) as pool:
        hash_results = list(pool.map(cache.get_or_compute, group))
    
    # Détection des doublons
    removed = 0
    processed = set()
    
    for j, (path_a, hash_a) in enumerate(zip(group, hash_results)):
        if not hash_a or path_a in processed:
            continue
            
        duplicates = [path_a]
        
        for k, (path_b, hash_b) in enumerate(zip(group, hash_results)):
            if (k <= j or not hash_b or path_b in processed):
                continue
            
            try:
                if (hash_a.hash.shape == hash_b.hash.shape and 
                    (hash_a - hash_b) <= THRESHOLD):
                    duplicates.append(path_b)
                    processed.add(path_b)
            except Exception:
                continue
        
        # Suppression des doublons
        if len(duplicates) > 1:
            duplicates.sort(key=lambda p: p.name)
            for dup in duplicates[1:]:
                try:
                    dup.unlink()
                    removed += 1
                    logging.info(f"Supprimé {dup.parent.name}/{dup.name} (doublon de {duplicates[0].parent.name}/{duplicates[0].name})")
                except Exception as e:
                    logging.error(f"Échec suppression {dup}: {e}")
    
    return removed


def stable_dedupe(images: list[Path], cache: TurboCache) -> None:
    """Dédoublonnage stable et rapide pour CPU haute performance."""
    removed = 0
    
    # Statistiques par dossier
    folder_stats = defaultdict(int)
    for img in images:
        folder_stats[img.parent.name] += 1
    
    logging.info(f"Images par dossier: {dict(folder_stats)}")
    logging.info(f"Mode stable avec {WORKERS} workers")

    # Groupement par taille exacte
    size_groups = defaultdict(list)
    for img in images:
        try:
            size_groups[img.stat().st_size].append(img)
        except Exception:
            continue
    
    exact_size_candidates = [group for group in size_groups.values() if len(group) > 1]
    
    # Groupement par fenêtre pour les images uniques
    single_images = [group[0] for group in size_groups.values() if len(group) == 1]
    window_groups = defaultdict(list)
    for img in single_images:
        try:
            window_groups[img.stat().st_size // SIZE_WINDOW].append(img)
        except Exception:
            continue
    
    window_candidates = [group for group in window_groups.values() if len(group) > 1]
    
    all_candidates = exact_size_candidates + window_candidates
    total_candidates = sum(len(group) for group in all_candidates)
    
    if not all_candidates:
        logging.info("Aucun doublon potentiel détecté")
        return
    
    logging.info(f"{total_candidates} images candidates dans {len(all_candidates)} groupes")
    
    # Traitement stable par batches
    batch_count = 0
    for i in range(0, len(all_candidates), BATCH_SIZE):
        batch_groups = all_candidates[i:i + BATCH_SIZE]
        batch_count += 1
        
        logging.info(f"Batch {batch_count} - Traitement de {len(batch_groups)} groupes")
        
        # Traitement avec ThreadPoolExecutor uniquement (plus stable)
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = [executor.submit(process_group, group, cache) for group in batch_groups]
            
            # Collecte des résultats
            for i, future in enumerate(futures):
                try:
                    group_removed = future.result()
                    removed += group_removed
                    
                    if (i + 1) % 100 == 0:
                        logging.info(f"  Progression: {i+1}/{len(futures)} groupes traités")
                        
                except Exception as e:
                    logging.error(f"Erreur traitement groupe: {e}")
    
    logging.info(f"🚀 {removed} doublons supprimés en mode stable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dédoublonneur stable haute performance")
    parser.add_argument("--directory", type=Path, 
                       default=Path(r"T:\_SELECT\TODO\Kanpekiseijo"),
                       help="Répertoire à analyser")
    parser.add_argument("--clear-cache", action="store_true", 
                       help="Efface le cache avant de commencer")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    start = datetime.now()
    
    logging.info(f"🔥 Mode stable haute-performance - {WORKERS} workers pour 5950X")
    
    if args.clear_cache and CACHE_FILE.exists():
        CACHE_FILE.unlink()
        logging.info("Cache effacé")
    
    # Collecte des images
    logging.info("Scan du répertoire...")
    images = []
    for p in args.directory.rglob("*"):
        if (p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"} and 
            p.is_file()):
            try:
                if p.stat().st_size > 1024:
                    images.append(p)
            except Exception:
                continue
    
    cache = TurboCache()
    logging.info(f"🚀 Analyse stable de {len(images)} images")
    
    stable_dedupe(images, cache)
    
    cache.save()
    
    duration = (datetime.now() - start).total_seconds()
    if duration > 0:
        rate = len(images) / duration
        logging.info(f"✅ Terminé en {duration:.1f}s ({rate:.0f} img/s) - Mode stable")
    else:
        logging.info("✅ Terminé instantanément")


if __name__ == "__main__":
    main()