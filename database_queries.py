import sqlite3
from datetime import datetime

# ✅ Fetch product details by name
def get_product_details(product_name):
    """Fetch product details, ensuring all required fields are returned."""
    conn = sqlite3.connect("products.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, name, category, genre, price, description, image_path 
        FROM products 
        WHERE name = ?
    """, (product_name,))

    product = cursor.fetchone()
    conn.close()

    if product is None:
        return None  # Ensure function does not return empty tuple
    
    return product  # Should always return exactly 7 fields


# ✅ Update product interaction (held or pointed at)
def update_product_interaction(product_name, interaction_type):
    """
    Updates the `product_analytics` table to log interactions.
    - `interaction_type` can be either "held" or "pointed".
    """
    conn = sqlite3.connect("products.db")
    cursor = conn.cursor()

    # Get product ID
    cursor.execute("SELECT id FROM products WHERE name = ?", (product_name,))
    product = cursor.fetchone()

    if product:
        product_id = product[0]

        # Check if the product already has an entry in product_analytics
        cursor.execute("SELECT * FROM product_analytics WHERE product_id = ?", (product_id,))
        analytics_data = cursor.fetchone()

        if analytics_data:
            # Update interaction count
            if interaction_type == "held":
                cursor.execute("UPDATE product_analytics SET times_held = times_held + 1, last_updated = ? WHERE product_id = ?", (datetime.now(), product_id))
            elif interaction_type == "pointed":
                cursor.execute("UPDATE product_analytics SET times_pointed = times_pointed + 1, last_updated = ? WHERE product_id = ?", (datetime.now(), product_id))
        else:
            # Create a new entry if product doesn't exist in analytics
            if interaction_type == "held":
                cursor.execute("INSERT INTO product_analytics (product_id, times_held, last_updated) VALUES (?, 1, ?)", (product_id, datetime.now()))
            elif interaction_type == "pointed":
                cursor.execute("INSERT INTO product_analytics (product_id, times_pointed, last_updated) VALUES (?, 1, ?)", (product_id, datetime.now()))

        conn.commit()

    conn.close()


# ✅ Fetch analytics for managers (Full product stats)
def get_product_analytics():
    """
    Retrieves full product analytics for managers, including interaction data.
    Returns a list of all product analytics records.
    """
    conn = sqlite3.connect("products.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT p.name, pa.stock_quantity, pa.times_held, pa.times_pointed, pa.last_updated 
        FROM product_analytics pa
        JOIN products p ON pa.product_id = p.id
    """)

    data = cursor.fetchall()
    conn.close()
    
    return data


# ✅ Update stock quantity
def update_stock(product_name, new_stock_quantity):
    """
    Updates the stock quantity for a given product.
    """
    conn = sqlite3.connect("products.db")
    cursor = conn.cursor()

    # Get product ID
    cursor.execute("SELECT id FROM products WHERE name = ?", (product_name,))
    product = cursor.fetchone()

    if product:
        product_id = product[0]
        cursor.execute("UPDATE product_analytics SET stock_quantity = ? WHERE product_id = ?", (new_stock_quantity, product_id))
        conn.commit()

    conn.close()


# ✅ Get most popular products (based on interactions)
def get_most_popular_products(limit=5):
    """
    Retrieves the most popular products based on the sum of `times_held` and `times_pointed`.
    Returns the top `limit` products.
    """
    conn = sqlite3.connect("products.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT p.name, pa.times_held + pa.times_pointed AS total_interactions
        FROM product_analytics pa
        JOIN products p ON pa.product_id = p.id
        ORDER BY total_interactions DESC
        LIMIT ?
    """, (limit,))

    data = cursor.fetchall()
    conn.close()
    
    return data


# ✅ Fetch low-stock alerts (for managers)
def get_low_stock_alert(threshold=5):
    """
    Retrieves products where stock is below a given threshold.
    Default threshold is 5.
    """
    conn = sqlite3.connect("products.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT p.name, pa.stock_quantity
        FROM product_analytics pa
        JOIN products p ON pa.product_id = p.id
        WHERE pa.stock_quantity < ?
    """, (threshold,))

    data = cursor.fetchall()
    conn.close()
    
    return data
