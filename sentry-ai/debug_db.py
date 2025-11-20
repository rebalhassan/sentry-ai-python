# debug_db.py
"""Debug why tables aren't being created"""

import sqlite3

def test_direct_sqlite():
    """Test SQLite directly without our wrapper"""
    print("🔍 Testing direct SQLite...\n")
    
    # Test 1: In-memory database
    print("TEST 1: Creating in-memory database")
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO test VALUES (1, 'Billy')")
    
    result = conn.execute("SELECT * FROM test").fetchone()
    print(f"✅ In-memory DB works: {result}")
    conn.close()
    
    # Test 2: Our Database class
    print("\nTEST 2: Testing our Database class")
    from sentry.core.database import Database
    
    db = Database(db_path=":memory:")
    print(f"   db_path type: {type(db.db_path)}")
    print(f"   db_path value: {db.db_path}")
    
    # Try to query sources table
    try:
        with db.get_connection() as conn:
            tables = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table'
            """).fetchall()
            print(f"   Tables found: {[t[0] for t in tables]}")
            
            if not tables:
                print("   ❌ NO TABLES FOUND!")
                print("   Let's manually create one...")
                conn.execute("CREATE TABLE test2 (id INTEGER)")
                conn.commit()
                
                tables2 = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table'
                """).fetchall()
                print(f"   After manual create: {[t[0] for t in tables2]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    test_direct_sqlite()