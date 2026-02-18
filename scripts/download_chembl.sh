#!/bin/bash
#
# Download and extract ChEMBL SQLite database
#
# Usage: bash scripts/download_chembl.sh [VERSION]
#
# Example: bash scripts/download_chembl.sh 35
#

set -e

# Default to version 35, but allow override via command line
CHEMBL_VERSION="${1:-35}"
CHEMBL_BASE_URL="https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases"
CHEMBL_URL="${CHEMBL_BASE_URL}/chembl_${CHEMBL_VERSION}/chembl_${CHEMBL_VERSION}_sqlite.tar.gz"
DATA_DIR="data"

echo "==================================="
echo "ChEMBL SQLite Database Downloader"
echo "==================================="
echo

# Create data directory
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

TARBALL="chembl_${CHEMBL_VERSION}_sqlite.tar.gz"

# Check if already downloaded
if [ -f "$TARBALL" ]; then
    echo "✓ Found existing $TARBALL, skipping download"
else
    echo "Downloading ChEMBL ${CHEMBL_VERSION} SQLite database..."
    echo "URL: $CHEMBL_URL"
    echo "Size: ~4-5 GB (compressed), ~15 GB (extracted)"
    echo
    echo "Note: If download fails, check available versions at:"
    echo "  https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/"
    echo

    # Download with progress
    if command -v wget &> /dev/null; then
        wget --progress=bar:force "$CHEMBL_URL" -O "$TARBALL"
    elif command -v curl &> /dev/null; then
        curl -L -o "$TARBALL" "$CHEMBL_URL"
    else
        echo "Error: Neither wget nor curl found. Please install one of them."
        exit 1
    fi
fi

echo
echo "Extracting database..."
tar -xzf "$TARBALL"

# Find the .db file (structure is chembl_XX/chembl_XX_sqlite/chembl_XX.db)
DB_FILE=$(find chembl_${CHEMBL_VERSION} -name "*.db" 2>/dev/null | head -n 1)

if [ -z "$DB_FILE" ]; then
    echo "Error: Could not find .db file in extracted archive"
    exit 1
fi

echo
echo "✓ Database ready: $DATA_DIR/$DB_FILE"
echo
echo "To use with 01_fetch_smiles.py:"
echo "  python scripts/01_fetch_smiles.py --db $DATA_DIR/$DB_FILE"
echo
echo "Keeping tarball for future use: $TARBALL"

echo "✓ Done!"
