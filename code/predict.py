import re
import joblib
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

# -----------------------------
# Load Model & Vectorizer Bundle
# -----------------------------


model = joblib.load("svm_model.pltk")      
tfidf = joblib.load("vectorizer.pltk")    

# -----------------------------
# Noise Detection
# -----------------------------
def is_noise(text):
    if not isinstance(text, str):
        return True

    text = text.strip()


    if len(text) < 3:
        return True

    if not re.search(r"[a-zA-Z]", text):
        return True

    return False


# -----------------------------
# SINGLE INPUT MODE
# -----------------------------
def single_input_mode():
    print("\n🔹 Single Complaint Mode (type exit to stop)")

    while True:
        text = input("\nEnter Complaint: ")

        if text.lower() == "exit":
            break

        if is_noise(text):
            print("Predicted Category: Others")
            continue

        X = tfidf.transform([text])

        if X.nnz == 0:
            print("Predicted Category: Others")
            continue

        pred = model.predict(X)[0]
        print("Predicted Category:", pred)


# -----------------------------
# CSV BATCH MODE
# -----------------------------
def csv_input_mode():
    # Initialize Tkinter root
    root = tk.Tk()
    root.withdraw()  # Hide the main tiny tk window
    
    # --- FIX: Force dialog to the front ---
    root.lift()
    root.attributes("-topmost", True)
    
    # 1. Ask for Input File
    input_file = filedialog.askopenfilename(
        parent=root,
        title="Select Uncategorized Complaints CSV",
        filetypes=[("CSV Files", "*.csv")]
    )

    if not input_file:
        print("❌ No input file selected!")
        root.destroy()
        return

    # 2. Process the CSV
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read CSV:\n{e}", parent=root)
        root.destroy()
        return

    if "Complaint" not in df.columns:
        messagebox.showerror("Error", "CSV must contain a 'Complaint' column", parent=root)
        root.destroy()
        return

    print(f"Processing {len(df)} rows...")
    predictions = []

    for text in df["Complaint"]:
        # Handle empty/noise rows
        if is_noise(text):
            predictions.append("Others")
            continue

        # Transform and Predict
        X = tfidf.transform([str(text)])
        
        if X.nnz == 0:
            predictions.append("Others")
            continue

        pred = model.predict(X)[0]
        predictions.append(pred)

    df["Predicted_Category"] = predictions

    # 3. Ask for Output Location
    output_file = filedialog.asksaveasfilename(
        parent=root,
        title="Save Categorized Output CSV",
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv")]
    )

    if not output_file:
        print("⚠️ Output save cancelled. Results not saved.")
    else:
        df.to_csv(output_file, index=False)
        messagebox.showinfo("Success", f"✅ Categorized successfully!\nSaved to: {output_file}", parent=root)
        print(f"✅ Success! File saved at: {output_file}")

    # Clean up the Tkinter instance
    root.destroy()

# -----------------------------
# MAIN MENU
# -----------------------------
print("✅ Model Loaded Successfully!")
print("\nChoose Input Mode:")
print("1️⃣  Single Complaint Input")
print("2️⃣  CSV File Input")

choice = input("\nEnter choice (1 or 2): ").strip()

if choice == "1":
    single_input_mode()
elif choice == "2":
    csv_input_mode()
else:
    print("❌ Invalid choice. Please restart and choose 1 or 2.")
