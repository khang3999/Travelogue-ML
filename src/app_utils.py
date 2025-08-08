import json
import os
from firebase_admin import db as firebase_db, storage
from datetime import date
from sklearn.ensemble import RandomForestClassifier
import joblib
from tkinter import messagebox


# LẤY DATA ĐỂ TRAINING TỪ FIREBASE
def get_data_tour_training():
    tour_ref = firebase_db.reference("dataTourTraining/")
    data = tour_ref.get()
    if not data:
        print("No data found from firebase")
        return None
    # print(data)
    today = date.today()

    # Lưu tất cả file json
    folder_path = "json_files"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/{today}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # Lưu file json mới nhất dùng để train
    folder_path = "latest_json_file"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/latest.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return data


def delete_data_tour_training():
    tour_ref = firebase_db.reference("dataTourTraining/")
    data = tour_ref.get()
    if not data:
        messagebox.showinfo("Thông báo", "Không có dữ liệu để xóa.")
        return

    confirm = messagebox.askyesno(
        "Xác nhận", "Bạn có chắc muốn xóa toàn bộ dữ liệu của node dataTourTraining trên Firebase không?"
    )
    if confirm:
        # tour_ref.delete()
        messagebox.showinfo("Thông báo", "Đã xóa dữ liệu.")


# LOAD MODEL
def load_json_data(json_path):
    """Load file JSON nếu tồn tại"""
    if not os.path.exists(json_path):
        print(f"[WARN] File không tồn tại: {json_path}")
        return None  # hoặc None, tùy bạn muốn trả về gì

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[ERROR] File không phải JSON hợp lệ: {json_path}")
        return None


# LƯU MODEL
def save_model(model: RandomForestClassifier):
    try:
        folder_path = "models"
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, "tour_model.pkl")
        # Lưu lại model ở local
        joblib.dump(model, file_path)

        # Upload lên Firebase Storage
        bucket = storage.bucket()
        blob = bucket.blob("models/tour_model.pkl")  # đường dẫn trên Storage
        blob.upload_from_filename(file_path, content_type="application/octet-stream")
        print("Model saved to local and uploaded to Firebase Storage successfully.")
        return True
    except Exception as e:
        print(f"Error saving or uploading model: {e}")
        return False


# Xử lí dữ liệu trùng lặp và bỏ id
def remove_duplicates(data_json):
    unique_data = []
    seen = set()

    for item in data_json.values():
        # Chuyển item thành chuỗi JSON để có thể hash (so sánh dựa trên nội dung)
        item_str = json.dumps(item, sort_keys=True)
        if item_str not in seen:
            seen.add(item_str)
            unique_data.append(item)

    return unique_data
