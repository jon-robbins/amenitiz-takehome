#!/usr/bin/env python3
"""
Simple script to demonstrate how to explore the hotel data using the db module.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import the db module
sys.path.append(str(Path(__file__).parent))

from utils.db import init_db

def main():
    # Initialize the database
    print("Initializing database...")
    con = init_db()
    
    # Show available tables
    print("\nAvailable tables:")
    tables = con.execute("SHOW TABLES").fetchall()
    for table in tables:
        print(f"  - {table[0]}")
    
    # Show schema for each table
    print("\nTable schemas:")
    for table in tables:
        table_name = table[0]
        print(f"\n{table_name}:")
        schema = con.execute(f"DESCRIBE {table_name}").fetchall()
        for column in schema:
            print(f"  {column[0]}: {column[1]}")
    
    # Show sample data from each table
    print("\nSample data:")
    for table in tables:
        table_name = table[0]
        print(f"\n{table_name} (first 3 rows):")
        sample = con.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchdf()
        print(sample)
        print("-" * 50)

if __name__ == "__main__":
    main()