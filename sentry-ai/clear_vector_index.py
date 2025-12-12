# clear_vector_index.py
"""
Helper script to clear the vector index when changing embedding models.

Run this when you get dimension mismatch errors after upgrading the embedding model.
"""

import sys
from pathlib import Path

def clear_index():
    """Clear the FAISS vector index and metadata"""
    
    # Default path
    data_dir = Path.home() / ".sentry-ai"
    index_path = data_dir / "vectors.faiss"
    meta_path = data_dir / "vectors.faiss.meta"
    
    print("🧹 Clearing vector index...")
    print(f"   Data directory: {data_dir}")
    
    deleted = False
    
    # Delete index file
    if index_path.exists():
        index_path.unlink()
        print(f"   ✅ Deleted: {index_path}")
        deleted = True
    else:
        print(f"   ⚠️  Not found: {index_path}")
    
    # Delete metadata file
    if meta_path.exists():
        meta_path.unlink()
        print(f"   ✅ Deleted: {meta_path}")
        deleted = True
    else:
        print(f"   ⚠️  Not found: {meta_path}")
    
    if deleted:
        print("\n✅ Vector index cleared!")
        print("   You can now re-index your content with the new embedding model.")
    else:
        print("\n⚠️  No index files found. Nothing to delete.")
    
    return 0

if __name__ == "__main__":
    sys.exit(clear_index())
