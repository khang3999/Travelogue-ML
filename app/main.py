import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tkinter as tk
from src.app_utils import (
    get_data_tour_training,
    load_json_data,
    save_model,
    remove_duplicates,
    delete_data_tour_training
)
from src.ai_model import train_model
from database.firebase_config import init_firebase


init_firebase()

# scaler = joblib.load("models/tour_scaler.pkl")

# Tạo cửa sổ chính
window = tk.Tk()
window.title("Training model")
window.geometry("400x200")

# Action result
label_result = tk.Label(window, text="Action: ...", font=("Arial", 13))
label_result.pack(pady=5)

# CRAWL DATA ------------------------
frame_crawl = tk.Frame(window)
frame_crawl.pack(pady=5)

## Label
label_crawl = tk.Label(frame_crawl, text="Crawl data", font=("Arial", 11))
label_crawl.pack(side=tk.LEFT, padx=5)


## Handle
def handle_on_click_to_crawl_data():
    result = get_data_tour_training()
    textResult = (
        "Download and save files successfully"
        if result
        else "Download and save files failed"
    )
    label_result.config(text=f"Crawl: {textResult}")


## Button
button_crawl = tk.Button(
    frame_crawl, text="Crawl", command=handle_on_click_to_crawl_data
)
button_crawl.pack(side=tk.LEFT, padx=0)
# ------------ END CRAWL DATA

# TRAINING MODEL -------------------------
frame_train = tk.Frame(window)
frame_train.pack(pady=5)

## Label
label_train = tk.Label(frame_train, text="Train model", font=("Arial", 11))
label_train.pack(side=tk.LEFT, padx=5)

## Handle
def handle_on_click_to_training_model():
    # Load json
    file_path = os.path.join("latest_json_file", "latest.json")
    latest_json = load_json_data(file_path)

    # Xử lí thành mảng bỏ dữ liệu trùng lặp
    data_set = remove_duplicates(latest_json)
    print(data_set)
    # -- TRAINING --
    new_model = train_model(data_set)

    result = save_model(new_model)
    label_result.config(text=f"Train: {'Successfully' if result else 'Failed'}")

## Button
button_train = tk.Button(
    frame_train, text="Train", command=handle_on_click_to_training_model
)
button_train.pack(pady=0)
# -------------- END TRAINING MODEL

# DELETE DATA FIREBASE ----------------------
frame_delete = tk.Frame(window)
frame_delete.pack(pady=5)

## Label
label_delete = tk.Label(frame_delete, text="Clear data training", font=("Arial", 11))
label_delete.pack(side=tk.LEFT, padx=5)

## Handle
def handle_clear_data_training_firebase():
    delete_data_tour_training()

## Button
button_delete = tk.Button(
    frame_delete, text="Clear", command=handle_clear_data_training_firebase
)
button_delete.pack(pady=0)
# ----------- END DELETE DATA FIREBASE

# Chạy ứng dụng
window.mainloop()
