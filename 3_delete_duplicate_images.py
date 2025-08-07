#!/usr/bin/env python3
"""Detecteur/suppresseur de doublons d'images optimise anti-freeze."""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import lz4.frame

import imagehash
from PIL import Image

# Configuration anti-freeze pour Windows
HASH_SIZE = 16
SIZE_WINDOW = 1024 * 50
THRESHOLD = 1
CACHE_FILE = Path("image_hashes_cache.lz4")
# Configuration Windows-friendly
WORKERS = min(32, os.cpu_count() * 2)  # Limite a 32 workers max
BATCH_SIZE = 500  # Batches plus petits pour eviter les freeze
MAX_GROUP_SIZE = 100  # Limite la taille des groupes
CACHE_VERSION = "v6"

def compute_hash_precise(path: Path) -> tuple[Path, imagehash.ImageHash] | None:
    """Hash precise avec average_hash optimise."""
    try:
        with Image.open(path) as img:
            h = imagehash.average_hash(img, hash_size=HASH_SIZE)
            return path, h
    except Exception as e:
        logging.warning(f"Erreur {path}: {e}")
        return None

class TurboCache:
    """Cache compresse en memoire avec chargement lazy."""
    
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
                    logging.info("Version de cache obsolete, reconstruction necessaire")
                    return
                
                self.data = data.get('hashes', {})
                self.stats = data.get('stats', {})
                logging.info(f"Cache charge: {len(self.data)} entrees")
        except Exception as e:
            logging.warning(f"Erreur chargement cache: {e}")
    
    def save(self) -> None:
        """Sauvegarde compressee ultra-rapide."""
        try:
            data = {
                'version': self.version,
                'hashes': self.data, 
                'stats': self.stats
            }
            compressed = lz4.frame.compress(pickle.dumps(data))
            CACHE_FILE.write_bytes(compressed)
            logging.info(f"Cache sauve: {len(self.data)} entrees")
        except Exception as e:
            logging.error(f"Erreur sauvegarde cache: {e}")
    
    def is_fresh(self, path: Path) -> bool:
        """Verifie si le hash est encore valide."""
        path_str = str(path)
        try:
            return (path_str in self.data and 
                    path_str in self.stats and 
                    self.stats[path_str] == path.stat().st_mtime)
        except Exception:
            return False
    
    def get_or_compute(self, path: Path) -> imagehash.ImageHash | None:
        """Recupere ou calcule le hash d'une image."""
        path_str = str(path)
        
        # Verifier le cache
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

def process_group_safe(group: List[Path], cache: TurboCache) -> int:
    """Traite un groupe d'images de maniere securisee pour Windows."""
    if len(group) < 2:
        return 0
    
    # Log pour gros groupes
    if len(group) > 20:
        logging.info(f"Traitement groupe de {len(group)} images...")
    
    # Diviser les gros groupes
    if len(group) > MAX_GROUP_SIZE:
        logging.info(f"Groupe enorme ({len(group)} images), division en sous-groupes")
        removed = 0
        for i in range(0, len(group), MAX_GROUP_SIZE):
            subgroup = group[i:i + MAX_GROUP_SIZE]
            removed += process_group_safe(subgroup, cache)
        return removed
    
    # Hash avec nombre limite de workers
    max_workers = min(16, len(group))  # Max 16 workers par groupe
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        hash_results = list(pool.map(cache.get_or_compute, group))
    hash_time = time.time() - start_time
    
    if hash_time > 2:  # Si plus de 2 secondes
        logging.info(f"Hash de {len(group)} images: {hash_time:.1f}s")
    
    # Detection des doublons
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
        
        # Suppression sequentielle pour eviter les conflits Windows
        if len(duplicates) > 1:
            duplicates.sort(key=lambda p: p.name)
            
            if len(duplicates) > 10:
                logging.info(f"Trouve {len(duplicates)} doublons de {duplicates[0].name}")
            
            for dup in duplicates[1:]:
                try:
                    dup.unlink()
                    removed += 1
                    logging.info(f"Supprime {dup.parent.name}/{dup.name} (doublon de {duplicates[0].parent.name}/{duplicates[0].name})")
                    
                    # Petite pause pour Windows apres suppressions multiples
                    if removed % 50 == 0:
                        time.sleep(0.01)
                        
                except Exception as e:
                    logging.error(f"Echec suppression {dup}: {e}")
    
    return removed

def windows_friendly_dedupe(images: list[Path], cache: TurboCache) -> None:
    """Dedoublonnage optimise pour Windows."""
    removed = 0
    
    # Statistiques par dossier
    folder_stats = defaultdict(int)
    for img in images:
        folder_stats[img.parent.name] += 1
    
    logging.info(f"Images par dossier: {dict(folder_stats)}")
    logging.info(f"Mode Windows-friendly avec {WORKERS} workers max")

    # Groupement par taille exacte
    size_groups = defaultdict(list)
    for img in images:
        try:
            size_groups[img.stat().st_size].append(img)
        except Exception:
            continue
    
    exact_size_candidates = [group for group in size_groups.values() if len(group) > 1]
    
    # Groupement par fenetre pour les images uniques
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
        logging.info("Aucun doublon potentiel detecte")
        return
    
    # Trier par taille de groupe (petits d'abord)
    all_candidates.sort(key=len)
    
    logging.info(f"{total_candidates} images candidates dans {len(all_candidates)} groupes")
    
    # Traitement par batches avec timeout
    batch_count = 0
    for i in range(0, len(all_candidates), BATCH_SIZE):
        batch_groups = all_candidates[i:i + BATCH_SIZE]
        batch_count += 1
        
        logging.info(f"Batch {batch_count}/{(len(all_candidates) + BATCH_SIZE - 1) // BATCH_SIZE} - {len(batch_groups)} groupes")
        
        # Traitement avec gestion d'erreurs mais sans timeout global
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(process_group_safe, group, cache): group for group in batch_groups}
            
            # Collecte sans timeout (mais avec timeout individuel)
            completed = 0
            for future in as_completed(futures):
                try:
                    group_removed = future.result(timeout=300)  # 5 minutes max par groupe
                    removed += group_removed
                    completed += 1
                    
                    # Progression toutes les 50 taches
                    if completed % 50 == 0:
                        logging.info(f"  Progression: {completed}/{len(futures)} groupes traites")
                        
                except Exception as e:
                    group = futures[future]
                    logging.error(f"Erreur groupe de {len(group)} images: {e}")
                    completed += 1
        
        # Petite pause entre batches pour laisser Windows respirer
        if batch_count % 3 == 0:  # Pause plus frequente
            logging.info("Pause technique...")
            time.sleep(0.2)

    logging.info(f"{removed} doublons supprimes (mode Windows-friendly)")

def main() -> None:
    parser = argparse.ArgumentParser(description="Dedoublonneur Windows-friendly")
    parser.add_argument("--directory", type=Path, 
                       default=Path(r"T:\_SELECT\TODO\Danjo no Yuujou wa Seiritsu suru"),
                       help="Repertoire a analyser")
    parser.add_argument("--clear-cache", action="store_true", 
                       help="Efface le cache avant de commencer")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    start = datetime.now()
    
    logging.info(f"Mode Windows-friendly - {WORKERS} workers max")
    
    if args.clear_cache and CACHE_FILE.exists():
        CACHE_FILE.unlink()
        logging.info("Cache efface")
    
    # Collecte des images
    logging.info("Scan du repertoire...")
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
    logging.info(f"Analyse Windows-friendly de {len(images)} images")
    
    windows_friendly_dedupe(images, cache)
    
    cache.save()
    
    duration = (datetime.now() - start).total_seconds()
    if duration > 0:
        rate = len(images) / duration
        logging.info(f"Termine en {duration:.1f}s ({rate:.0f} img/s) - Mode Windows")
    else:
        logging.info("Termine instantanement")

if __name__ == "__main__":
    main()
