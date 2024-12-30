#!/usr/bin/env python3

import time
import serial
import joblib
import pandas as pd
import warnings

# ==============================
# Configurations
# ==============================
SERIAL_PORT = 'COM4'             # Update to your actual ESP32 port, e.g. '/dev/ttyUSB0'
BAUD_RATE = 115200
BATCH_SIZE = 15
STALE_THRESHOLD_SECONDS = 15      # If batch is older than 15s by the time we gather 15 lines, discard

# Paths to your models
MODELS = {
    "XGBoost_WithTime": "BrainScan_XGBoost_WithTime_12262024.pkl",
    "XGBoost_WithoutTime": "BrainScan_XGBoost_WithoutTime_12262024.pkl",
    "GradientBoosting_WithTime": "BrainScan_GradientBoosting_WithTime_12262024.pkl",
    "RandomForest_WithTime": "BrainScan_RandomForest_WithTime_12262024.pkl",
    "MLP_WithoutTime": "BrainScan_MLP_WithoutTime.pkl",
}

# The columns (features) your models expect
FEATURE_COLUMNS = [f"A{i}" for i in range(1, 16)] + [f"B{i}" for i in range(1, 16)]

# ==============================
# Suppress Warnings
# (like the scikit-learn version warnings, user warnings, etc.)
# ==============================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ==============================
# Model Loading & Utility
# ==============================
def load_models(model_paths):
    """
    Load all pre-trained models from disk.
    """
    models = {}
    for name, path in model_paths.items():
        try:
            models[name] = joblib.load(path)
            print(f"[INFO] Loaded model: {name} from '{path}'")
        except Exception as e:
            print(f"[WARNING] Could not load model '{name}' from '{path}': {e}")
    return models


def parse_sensor_data(raw_line):
    """
    Parse sensor data from a single line of text (e.g. "A1:123.45,A2:67.89,B15:12.3,...").
    Preserves the weird 'B15' logic (truncating extra items, if any).
    Returns a dict with keys A1..A15, B1..B15 if found, else partial or empty.
    """
    sensor_values = {}
    parts = raw_line.split(',')
    for part in parts:
        part = part.strip()
        if not part or ':' not in part:
            # Not a valid "key:value" chunk
            continue
        try:
            key, value = part.split(':', 1)
            key = key.strip()
            value = value.strip()

            if key == "B15":
                # B15 special trimming
                value_list = value.split(',')
                if len(value_list) > 3:
                    value_list = value_list[:3]
                value_list = [v[:3] for v in value_list]
                value = ",".join(value_list)

            sensor_values[key] = value
        except ValueError:
            # Malformed chunk
            continue

    return sensor_values


def calculate_stability(predictions, true_label):
    """
    For each batch, checks how many consecutive predictions
    match the 'true_label' from the start.
    Returns (first_correct_index, stability_score).
    """
    first_correct = None
    sustained_correct = 0
    total = len(predictions)

    for i, pred in enumerate(predictions):
        if pred == true_label:
            sustained_correct += 1
            if first_correct is None:
                first_correct = i + 1  # 1-based index
        else:
            break

    stability_score = sustained_correct / total if total else 0
    return first_correct, stability_score


def measure_correctness(predictions, true_label):
    """
    Counts how many predictions match 'true_label' out of total in batch.
    Returns (correctness, correct_count, incorrect_count).
    correctness = correct_count / total
    """
    total = len(predictions)
    correct_count = sum(1 for p in predictions if p == true_label)
    incorrect_count = total - correct_count
    correctness = correct_count / total if total else 0
    return correctness, correct_count, incorrect_count


def run_models(models, df_features, true_label):
    """
    Runs each model on df_features (DataFrame),
    returning a dict with:
      - TimeToPredict
      - Stability
      - First Correct
      - Correctness
      - Correct Count
      - Incorrect Count
    or an "Error" key if something broke.
    """
    results = {}
    for model_name, model in models.items():
        try:
            start_pred = time.time()
            predictions = model.predict(df_features)
            end_pred = time.time()

            # Calculate stability
            first_correct, stability_score = calculate_stability(predictions, true_label)
            # Calculate correctness
            correctness, correct_count, incorrect_count = measure_correctness(predictions, true_label)

            results[model_name] = {
                "TimeToPredict (s)": end_pred - start_pred,
                "Stability": stability_score,
                "First Correct": first_correct,
                "Correctness": correctness,
                "Correct Count": correct_count,
                "Incorrect Count": incorrect_count
            }
        except Exception as e:
            results[model_name] = {"Error": str(e)}
    return results


# ==============================
# Live Scanning & Batching
# ==============================
def record_and_test_live(models):
    """
    1) Prompt user for a label.
    2) Continuously read lines from the serial port, ignoring lines that aren't sensor data.
    3) For every 15 valid sensor lines, check if they're "stale" (>15s old). If so, discard.
       Otherwise, parse them into floats, run each model, measure time & correctness.
    4) Ctrl+C => return to label prompt; 'exit' => quit entirely.
    """
    # Try opening the serial port
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[INFO] Connected to serial port: {SERIAL_PORT}")
    except serial.SerialException as e:
        print(f"[ERROR] Could not open serial port {SERIAL_PORT}: {e}")
        return

    while True:
        label = input("\nEnter the label for this live session (type 'exit' to quit): ").strip()
        if 'exit' in label.lower():
            print("[INFO] Exiting entire live scanning process.")
            break

        print(f"[INFO] Starting live scan for label '{label}'. Reading {BATCH_SIZE} lines per batch...\n")
        time.sleep(1)

        try:
            while True:
                batch_data = []
                batch_timestamps = []

                # Step 1: Accumulate BATCH_SIZE valid lines
                while len(batch_data) < BATCH_SIZE:
                    raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not raw_line:
                        # Possibly no data, wait a bit
                        time.sleep(0.02)
                        continue

                    # Debug print
                    print(f"[SERIAL] {raw_line}")

                    # Attempt to parse sensor data
                    parsed = parse_sensor_data(raw_line)
                    # If we got no keys that match A1..B15, it's not real sensor data => skip
                    if not any(k in parsed for k in FEATURE_COLUMNS):
                        continue

                    # We consider the time stamp now
                    line_time = time.time()

                    # Convert the string fields to float or 0.0
                    row_floats = []
                    for col in FEATURE_COLUMNS:
                        val_str = parsed.get(col, "")
                        try:
                            val_float = float(val_str)
                        except ValueError:
                            val_float = 0.0
                        row_floats.append(val_float)

                    batch_data.append(row_floats)
                    batch_timestamps.append(line_time)

                # Step 2: Check stale
                # If the oldest line is >15s behind "now", discard the entire batch
                oldest_ts = min(batch_timestamps) if batch_timestamps else time.time()
                now_ts = time.time()
                if (now_ts - oldest_ts) > STALE_THRESHOLD_SECONDS:
                    print("[WARNING] Batch is stale (>15s old). Discarding and starting fresh...")
                    continue  # Skip predictions, gather new batch

                # Step 3: Build DataFrame
                df_batch = pd.DataFrame(batch_data, columns=FEATURE_COLUMNS)
                # Step 4: Run the models
                start_eval = time.time()
                results = run_models(models, df_batch, label)
                end_eval = time.time()

                # Step 5: Print out results
                print(f"\n=== Results for the last {BATCH_SIZE}-line batch ===")
                print(f"Time to run ALL model predictions: {end_eval - start_eval:.4f} s")

                for model_name, outcome in results.items():
                    if "Error" in outcome:
                        print(f"- {model_name} -> ERROR: {outcome['Error']}")
                    else:
                        print(f"- {model_name}:\n"
                              f"   TimeToPredict: {outcome['TimeToPredict (s)']:.4f} s\n"
                              f"   Stability:     {outcome['Stability']:.3f}\n"
                              f"   FirstCorrect:  {outcome['First Correct']}\n"
                              f"   Correctness:   {outcome['Correctness']:.3f}\n"
                              f"   CorrectCount:  {outcome['Correct Count']}\n"
                              f"   IncorrectCount:{outcome['Incorrect Count']}\n")
                print("============================================\n")

        except KeyboardInterrupt:
            # If user hits Ctrl+C, break out to label prompt
            print("\n[INFO] Caught Ctrl+C, returning to label prompt...\n")
            continue

    ser.close()
    print("[INFO] Serial port closed.")


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    # 1) Load the models
    all_models = load_models(MODELS)

    # 2) Run the live record + test routine
    record_and_test_live(all_models)
