#!/usr/bin/env python3
"""
Fetch diverse SMILES from ChEMBL SQLite database.

This script connects to a local ChEMBL SQLite database and extracts
small molecule SMILES with molecular weight between 150-900 Da.

Download ChEMBL SQLite from:
https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/

Example:
    wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_34_sqlite.tar.gz
    tar -xzf chembl_34_sqlite.tar.gz
"""

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import Optional

from rdkit import Chem
from tqdm import tqdm


def fetch_smiles_from_chembl_sqlite(
    db_path: str,
    output_path: str,
    n: int = 20000,
    min_mw: float = 150.0,
    max_mw: float = 900.0,
) -> None:
    """
    Fetch SMILES from ChEMBL SQLite database.

    Args:
        db_path: Path to ChEMBL SQLite database file (.db or .sqlite)
        output_path: Path to output CSV file
        n: Number of SMILES to fetch
        min_mw: Minimum molecular weight (freebase)
        max_mw: Maximum molecular weight (freebase)
    """
    # Connect to database
    print(f"Connecting to {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # SQL query to fetch diverse small molecules
    query = """
        SELECT DISTINCT cs.canonical_smiles, md.chembl_id, cp.mw_freebase
        FROM compound_structures cs
        JOIN molecule_dictionary md ON cs.molregno = md.molregno
        JOIN compound_properties cp ON cs.molregno = cp.molregno
        WHERE cp.mw_freebase BETWEEN ? AND ?
          AND cs.canonical_smiles NOT LIKE '%.%'
          AND cs.canonical_smiles IS NOT NULL
          AND md.chembl_id IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """

    print(f"Fetching up to {n} SMILES (MW: {min_mw}–{max_mw} Da)...")
    cursor.execute(query, (min_mw, max_mw, n))
    rows = cursor.fetchall()
    conn.close()

    print(f"Retrieved {len(rows)} rows from database")

    # Validate with RDKit and collect
    collected = []
    seen = set()

    for smiles, chembl_id, mw in tqdm(rows, desc="Validating"):
        # Skip if already seen
        if smiles in seen:
            continue

        # Validate with RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # Canonicalize
        canonical = Chem.MolToSmiles(mol)
        if canonical in seen:
            continue

        seen.add(canonical)
        collected.append({
            'chembl_id': chembl_id,
            'smiles': canonical,
            'num_atoms': mol.GetNumHeavyAtoms(),
            'mw': round(mw, 2) if mw else None,
        })

    # Save to CSV
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        fieldnames = ['chembl_id', 'smiles', 'num_atoms', 'mw']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(collected)

    print(f"\n✓ Saved {len(collected)} valid SMILES to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch SMILES from ChEMBL SQLite database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Download ChEMBL database first:
    wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_34_sqlite.tar.gz
    tar -xzf chembl_34_sqlite.tar.gz
    
    # Then run this script:
    python scripts/01_fetch_smiles.py --db chembl_34/chembl_34_sqlite/chembl_34.db
        """
    )
    parser.add_argument(
        '--db',
        type=str,
        default='chembl_34.db',
        help='Path to ChEMBL SQLite database file (default: chembl_34.db)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/smiles/chembl_smiles.csv',
        help='Output CSV file path (default: data/smiles/chembl_smiles.csv)'
    )
    parser.add_argument(
        '--n',
        type=int,
        default=20000,
        help='Number of SMILES to fetch (default: 20000)'
    )
    parser.add_argument(
        '--min-mw',
        type=float,
        default=150.0,
        help='Minimum molecular weight (default: 150.0)'
    )
    parser.add_argument(
        '--max-mw',
        type=float,
        default=900.0,
        help='Maximum molecular weight (default: 900.0)'
    )

    args = parser.parse_args()

    # Check if database exists
    if not Path(args.db).exists():
        print(f"Error: Database file not found: {args.db}")
        print("\nPlease download ChEMBL SQLite database from:")
        print("https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/")
        print("\nExample commands:")
        print("  wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_34_sqlite.tar.gz")
        print("  tar -xzf chembl_34_sqlite.tar.gz")
        return 1

    fetch_smiles_from_chembl_sqlite(
        db_path=args.db,
        output_path=args.output,
        n=args.n,
        min_mw=args.min_mw,
        max_mw=args.max_mw,
    )

    return 0


if __name__ == "__main__":
    exit(main())
