import os
import sqlite3
from datetime import datetime
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Forces UTF-8 output
# Step 1: Ensure the database file is deleted before creating a new one
db_path = "products.db"

if os.path.exists(db_path):
    try:
        os.remove(db_path)
        print("✅ Old database deleted. Creating a new one...")
    except PermissionError:
        print("❌ Error: Database is locked! Close any processes using SQLite and try again.")
        exit()

# Step 2: Connect to SQLite (creates a fresh new database)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Step 3: Create the products table (including image path)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        genre TEXT NOT NULL,
        price REAL NOT NULL,
        description TEXT NOT NULL,
        image_path TEXT NOT NULL,   -- Ensure every product has an image
        brand TEXT,
        rating REAL DEFAULT 0.0,
        num_reviews INTEGER DEFAULT 0,
        availability_status TEXT DEFAULT 'In Stock',
        discount REAL DEFAULT 0.0,
        barcode TEXT UNIQUE
    )
''')

# Step 4: Insert products with correct image paths
products = [
    ("Teriaq-Intense", "Perfume", "Sweet", 180, 
     "A rich and captivating sweet fragrance with deep notes of vanilla and caramel, leaving a long-lasting impression.",
     r"C:\Users\Adonis\OneDrive\Desktop\DataScience\Projects\MotionDetection\StreamLit\Images\Teriaq-Intense.jpeg",
     "Luxury Scents", 4.5, 120, "In Stock", 10.0, "1234567890"),

    ("Afnan", "Perfume", "Masculine", 250, 
     "A bold and sophisticated masculine scent with woody and spicy notes, designed for confidence and elegance.",
     r"C:\Users\Adonis\OneDrive\Desktop\DataScience\Projects\MotionDetection\StreamLit\Images\Afnan-Not only intense.jpg",
     "Afnan", 4.7, 95, "In Stock", 5.0, "0987654321")
]

cursor.executemany("INSERT INTO products (name, category, genre, price, description, image_path, brand, rating, num_reviews, availability_status, discount, barcode) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", products)

# Step 5: Create the product_analytics table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS product_analytics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER NOT NULL,
        stock_quantity INTEGER DEFAULT 0,
        times_held INTEGER DEFAULT 0,
        times_pointed INTEGER DEFAULT 0,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        shelf_location TEXT,
        avg_interaction_time REAL DEFAULT 0.0,
        purchase_count INTEGER DEFAULT 0,
        restock_frequency INTEGER DEFAULT 0,
        FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
    )
''')

# Step 6: Insert mock analytics data
product_analytics = [
    (1, 50, 30, 20, datetime.now(), "Aisle 3 - Shelf B", 5.4, 10, 15),
    (2, 30, 45, 35, datetime.now(), "Aisle 1 - Shelf A", 6.2, 20, 10)
]

cursor.executemany("INSERT INTO product_analytics (product_id, stock_quantity, times_held, times_pointed, last_updated, shelf_location, avg_interaction_time, purchase_count, restock_frequency) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", product_analytics)

# Step 7: Commit changes and close the connection
conn.commit()
conn.close()

print("✅ New database created successfully!")

# Step 8: Verify Data - Fetch and display all products and analytics
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("\n📦 Stored Products:")
cursor.execute("SELECT * FROM products")
for row in cursor.fetchall():
    print(row)

print("\n📊 Product Analytics:")
cursor.execute("SELECT * FROM product_analytics")
for row in cursor.fetchall():
    print(row)

conn.close()
print("✅ Database setup completed successfully!")
