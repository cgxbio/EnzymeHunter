from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from scipy.spatial.distance import pdist, squareform
import biotite.structure as bs
from biotite.structure.io.pdb import PDBFile, get_structure
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="FASTA -> ContactMap (8Å from PDB) + ResNet101 embedding"
    )
    parser.add_argument(
        "--fasta-in", type=Path, required=True,
        help="Input FASTA file with protein IDs",
    )
    parser.add_argument(
        "--pdb-dir", type=Path, default=Path("./data/pdb"),
        help="Directory containing PDB files named {UNIPROT_ID}.pdb (default: ./data/pdb)",
    )
    parser.add_argument(
        "--out-conmap-dir", type=Path,
        default=Path("./data/datasets_process/contactmap_rep/contactmap_8a"),
        help="Output directory for contact map .npy files",
    )
    parser.add_argument(
        "--out-emb-dir", type=Path,
        default=Path("./data/datasets_process/contactmap_rep/contactmap_resnet101_rep"),
        help="Output directory for ResNet101 embedding .pt files",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="cuda / cuda:0 / cuda:1 / cpu",
    )
    return parser.parse_args()

def _extend(a, b, c, L, A, D):
    """Place a virtual Cβ atom using backbone N, CA, C coordinates."""
    def normalize(x):
        return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)

    bc = normalize(b - c)
    n = normalize(np.cross(b - a, bc))
    m = [bc, np.cross(n, bc), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    return c + sum([mi * di for mi, di in zip(m, d)])


def contacts_from_pdb(
    structure: bs.AtomArray,
    distance_threshold: float = 8.0,
    chain: Optional[str] = None,
) -> np.ndarray:

    mask = ~structure.hetero
    if chain is not None:
        mask &= (structure.chain_id == chain)

    N  = structure.coord[mask & (structure.atom_name == "N")]
    CA = structure.coord[mask & (structure.atom_name == "CA")]
    C  = structure.coord[mask & (structure.atom_name == "C")]

    if len(N) == 0 or len(CA) == 0 or len(C) == 0:
        raise ValueError("Missing backbone atoms (N/CA/C) in structure.")

    Cbeta = _extend(C, N, CA, 1.522, 1.927, -2.143)
    dist = squareform(pdist(Cbeta))

    contacts = (dist < distance_threshold).astype(np.int64)
    contacts[np.isnan(dist)] = -1
    return contacts

_uniprot_pat = re.compile(r"\b(?:sp|tr)\|([A-Z0-9]+)\|", re.IGNORECASE)


def read_fasta_ids(fp: Path) -> List[str]:
    """Read unique protein IDs from a FASTA file (order-preserving)."""
    ids: List[str] = []
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            h = line[1:].strip()
            if not h:
                continue

            m = _uniprot_pat.search(h)
            if m:
                uid = m.group(1).upper()
            else:
                uid = h.split()[0].split("|")[0].strip()
                uid = re.sub(r"[^A-Za-z0-9_.-]", "", uid).upper()

            if uid:
                ids.append(uid)

    seen: set = set()
    out: List[str] = []
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def read_structure_pdb_only(pdb_path: Path) -> bs.AtomArray:
    """Read a .pdb file and return the first model as AtomArray."""
    if pdb_path.suffix.lower() != ".pdb":
        raise ValueError(f"Only .pdb is supported, got: {pdb_path}")
    pdb_file = PDBFile.read(pdb_path)
    return get_structure(pdb_file)[0]


def load_resnet101(device: str) -> nn.Module:
    """
    Load ResNet101 with the FC layer replaced by Identity (feature extractor).
    Uses IMAGENET1K_V2 weights from torchvision.
    """
    model = torchvision.models.resnet101(weights="IMAGENET1K_V2")
    model.fc = nn.Identity()
    model = model.eval().to(device)
    return model


def contactmap_to_3ch_tensor(contact_map: np.ndarray) -> torch.Tensor:

    cmap3 = np.array([contact_map, contact_map, contact_map])
    return torch.tensor(cmap3).unsqueeze(0).to(dtype=torch.float32)

def process_pdb_contact_maps(
    fasta_path: str,
    pdb_dir: str = "./data/pdb",
    out_conmap_dir: str = "./data/datasets_process/contactmap_rep/contactmap_8a",
    out_emb_dir: str = "./data/datasets_process/contactmap_rep/contactmap_resnet101_rep",
    device: str = "cuda",
    distance_threshold: float = 8.0,
):

    fasta_path = Path(fasta_path)
    pdb_dir = Path(pdb_dir)
    out_conmap_dir = Path(out_conmap_dir)
    out_emb_dir = Path(out_emb_dir)
    out_conmap_dir.mkdir(parents=True, exist_ok=True)
    out_emb_dir.mkdir(parents=True, exist_ok=True)

    device = device if torch.cuda.is_available() else "cpu"

    # --- Read FASTA IDs ---
    prot_ids = read_fasta_ids(fasta_path)
    print(f"[INFO] FASTA IDs: {len(prot_ids)} from {fasta_path}")
    print(f"[INFO] PDB directory: {pdb_dir}")

    # --- Load ResNet101 ---
    model = load_resnet101(device=device)

    error_ids: List[str] = []
    error_lines: List[str] = []

    for uid in tqdm(prot_ids, desc="PDB contactmap + ResNet101"):
        try:
            npy_out = out_conmap_dir / f"{uid}.npy"
            pt_out = out_emb_dir / f"{uid}.pt"

            if npy_out.exists() and pt_out.exists():
                continue

            pdb_path = pdb_dir / f"{uid}.pdb"
            if not pdb_path.exists():
                error_ids.append(uid)
                error_lines.append(f"{uid}\tPDB_NOT_FOUND\t{pdb_path}")
                continue

            structure = read_structure_pdb_only(pdb_path)
            contact_map = contacts_from_pdb(
                structure, distance_threshold=distance_threshold
            )

            np.save(npy_out, contact_map)

            x = contactmap_to_3ch_tensor(contact_map).to(device)
            with torch.no_grad():
                emb = model(x).detach().cpu().squeeze(0)
            torch.save(emb, pt_out)

        except Exception as e:
            error_ids.append(uid)
            error_lines.append(f"{uid}\t{type(e).__name__}\t{e}")
            continue

    # --- Save error report ---
    if error_ids:
        err_file = out_conmap_dir / "error_ids.txt"
        err_file.write_text("\n".join(error_ids), encoding="utf-8")
        err_detail = err_file.with_suffix(".detail.txt")
        err_detail.write_text("\n".join(error_lines), encoding="utf-8")
        print(f"[DONE] errors: {len(error_ids)} → {err_file}")
    else:
        print("[DONE] All proteins processed successfully.")

def main():
    args = parse_args()
    process_pdb_contact_maps(
        fasta_path=str(args.fasta_in),
        pdb_dir=str(args.pdb_dir),
        out_conmap_dir=str(args.out_conmap_dir),
        out_emb_dir=str(args.out_emb_dir),
        device=args.device,
    )


if __name__ == "__main__":
    main()
