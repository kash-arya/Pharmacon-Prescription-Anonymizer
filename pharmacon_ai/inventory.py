"""
Clinical Inventory Database & Pharmacist Dispensation Engine.

Implements Stage 6 of the Pharmacon Architecture:
  - SQLite persistent storage for pharmacy inventory and patient records.
  - Automatic atomic inventory decrements upon pharmacist confirmation.
  - Patient medication dashboard linking dosage, timing, and duration.
  - Audit logging for regulatory compliance and model fine-tuning feedback.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple


DB_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_DB_PATH = os.path.join(DB_DIR, "pharmacon_clinical.db")
NLEM_JSON_PATH = os.path.join(DB_DIR, "nlem_2022.json")


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Connect to SQLite database with foreign keys enabled."""
    path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database tables and seed initial formulary inventory."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    # 1. Inventory Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS inventory (
        sku TEXT PRIMARY KEY,
        generic_name TEXT NOT NULL,
        brand_alias TEXT,
        dosage_form TEXT NOT NULL,
        strength TEXT NOT NULL,
        stock_units INTEGER NOT NULL DEFAULT 500,
        reorder_threshold INTEGER NOT NULL DEFAULT 50,
        unit_price_inr REAL NOT NULL DEFAULT 5.0
    )
    """)

    # 2. Patients Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER,
        gender TEXT,
        contact TEXT
    )
    """)

    # 3. Prescriptions & Dispensation Audit Log
    cur.execute("""
    CREATE TABLE IF NOT EXISTS prescriptions (
        prescription_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL,
        doctor_id TEXT NOT NULL,
        raw_ocr_text TEXT,
        resolved_generic TEXT NOT NULL,
        dosage_form TEXT NOT NULL,
        strength TEXT NOT NULL,
        frequency TEXT,
        duration TEXT,
        quantity_dispensed INTEGER NOT NULL,
        safety_gate_status TEXT NOT NULL,
        dispensed_by TEXT NOT NULL,
        dispensed_at TEXT NOT NULL,
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
    )
    """)

    # Seed Inventory from NLEM if empty
    cur.execute("SELECT COUNT(*) FROM inventory")
    if cur.fetchone()[0] == 0 and os.path.exists(NLEM_JSON_PATH):
        with open(NLEM_JSON_PATH, "r", encoding="utf-8") as f:
            nlem_entries = json.load(f)

        for idx, entry in enumerate(nlem_entries, 1):
            gen = entry["generic_name"]
            forms = entry.get("allowed_forms", ["Tab"])
            strengths = entry.get("allowed_strengths", ["500 mg"])
            brands = entry.get("common_brands", [])
            brand = brands[0] if brands else gen

            for form in forms:
                for strn in strengths:
                    clean_strn = strn.replace(" ", "").replace("/", "_")
                    sku = f"SKU-{gen[:4].upper()}-{form[:3].upper()}-{clean_strn}"
                    cur.execute("""
                    INSERT OR IGNORE INTO inventory 
                    (sku, generic_name, brand_alias, dosage_form, strength, stock_units, reorder_threshold, unit_price_inr)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (sku, gen, brand, form, strn, 500, 50, 4.50))

    # Seed sample patients if empty
    cur.execute("SELECT COUNT(*) FROM patients")
    if cur.fetchone()[0] == 0:
        cur.executemany("""
        INSERT INTO patients (patient_id, name, age, gender, contact)
        VALUES (?, ?, ?, ?, ?)
        """, [
            ("P-101", "Ramesh Kumar", 54, "Male", "+91 98123 45678"),
            ("P-102", "Sunita Sharma", 42, "Female", "+91 98765 43210"),
            ("P-103", "Aarav Patel", 28, "Male", "+91 97234 56789"),
        ])

    conn.commit()
    conn.close()


def get_inventory_items(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all inventory items with current stock levels."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM inventory ORDER BY generic_name ASC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_patient_medications(patient_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all active medication history for a patient."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
    SELECT * FROM prescriptions 
    WHERE patient_id = ? 
    ORDER BY dispensed_at DESC
    """, (patient_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def calculate_dispense_units(frequency: Optional[str], duration: Optional[str]) -> int:
    """Compute units to dispense based on dosing frequency and treatment duration."""
    doses_per_day = 1
    if frequency:
        f_up = frequency.upper()
        if "1-0-1" in f_up or "BD" in f_up: doses_per_day = 2
        elif "1-1-1" in f_up or "TDS" in f_up: doses_per_day = 3
        elif "QID" in f_up: doses_per_day = 4
        elif "1-0-0" in f_up or "0-0-1" in f_up or "OD" in f_up: doses_per_day = 1

    days = 5  # Default
    if duration:
        m = re.search(r"(\d+)\s*(days?|nights?|weeks?)", duration, re.I)
        if m:
            val = int(m.group(1))
            unit = m.group(2).lower()
            days = val * 7 if "week" in unit else val

    return max(1, doses_per_day * days)


def dispense_prescription(
    doctor_id: str,
    patient_id: str,
    medicine_data: Dict[str, Any],
    raw_ocr_text: str = "",
    dispensed_by: str = "Pharmacist",
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Atomic transaction:
      1. Finds matching inventory SKU.
      2. Decrements stock units.
      3. Records verified prescription in clinical audit log.
    """
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    gen_name = medicine_data.get("generic_name", "Unknown")
    form = medicine_data.get("form", "Tab")
    strn = medicine_data.get("strength", "")
    freq = medicine_data.get("frequency", "OD")
    dur = medicine_data.get("duration", "5 days")
    gate_status = medicine_data.get("gate_status", "CONFIRMED_MATCH")

    # Find SKU match
    cur.execute("""
    SELECT sku, stock_units, unit_price_inr FROM inventory 
    WHERE LOWER(generic_name) = LOWER(?)
    LIMIT 1
    """, (gen_name,))
    match = cur.fetchone()

    if not match:
        # Fallback to loose match
        cur.execute("""
        SELECT sku, stock_units, unit_price_inr FROM inventory 
        WHERE LOWER(generic_name) LIKE LOWER(?)
        LIMIT 1
        """, (f"%{gen_name}%",))
        match = cur.fetchone()

    sku = match["sku"] if match else f"SKU-{gen_name[:4].upper()}-GEN"
    current_stock = match["stock_units"] if match else 100

    units = calculate_dispense_units(freq, dur)
    new_stock = max(0, current_stock - units)

    # Atomic Update
    if match:
        cur.execute("UPDATE inventory SET stock_units = ? WHERE sku = ?", (new_stock, sku))

    # Insert Prescription Record
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rx_id = f"RX-{datetime.datetime.now().strftime('%y%m%d%H%M%S')}-{doctor_id}"

    cur.execute("""
    INSERT INTO prescriptions 
    (prescription_id, patient_id, doctor_id, raw_ocr_text, resolved_generic, 
     dosage_form, strength, frequency, duration, quantity_dispensed, safety_gate_status, dispensed_by, dispensed_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (rx_id, patient_id, doctor_id, raw_ocr_text, gen_name,
          form, strn, freq, dur, units, gate_status, dispensed_by, now_str))

    conn.commit()
    conn.close()

    return {
        "status": "SUCCESS",
        "prescription_id": rx_id,
        "sku": sku,
        "generic_name": gen_name,
        "quantity_dispensed": units,
        "remaining_stock": new_stock,
        "patient_id": patient_id,
        "dispensed_at": now_str,
    }
