# ml_training/app.py
from src.ai_model import predict_score, train_model
import joblib
import os
from database.firebase_config import init_firebase
from firebase_admin import storage


tours = [
    {"id": "tour1", "locations": ["10", "14", "17"], "rating": 4.5},
    {"id": "tour11", "locations": ["10", "14", "17"], "rating": 0},
    {"id": "tour12", "locations": ["10", "14", "17"], "rating": 5},
    {"id": "tour2", "locations": ["20", "22"], "rating": 2.8},
    {"id": "tour3", "locations": ["01", "10", "14", "62"], "rating": 3.2},
    {"id": "tour4", "locations": ["70", "72", "75"], "rating": 4.9},
    {"id": "tour5", "locations": ["48", "49", "51"], "rating": 1.5},
    {"id": "tour6", "locations": ["36", "37"], "rating": 3.7},
    {"id": "tour7", "locations": ["62", "64", "66", "68"], "rating": 2.9},
    {"id": "tour8", "locations": ["83", "84", "86"], "rating": 4.1},
    {"id": "tour9", "locations": ["33", "34", "35"], "rating": 3.0},
    {"id": "tour10", "locations": ["91", "93", "94"], "rating": 4.8},
]

# Danh sách địa điểm người dùng đã chọn hoặc từng đến
user_locations = ["10", "14", "62"]

def load_model_from_local(model_path: str, scaler_path: str):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler

# Huấn luyện mô hình--- KHI NÀO TRAIN LẠI MỚI MỞ RA
model, scaler, X, y = train_model(tours, user_locations)


# Lưu mô hình và scaler
# os.makedirs("models", exist_ok=True)
# joblib.dump(model, "./models/tour_model.pkl")
# joblib.dump(scaler, "./models/tour_scaler.pkl")

# Nếu cần đánh giá mô hình, có thể gọi hàm evaluate_model
# evaluate_model(model, X, y)

# TEST
# Sắp xếp tour theo độ phù hợp giảm dần
# sorted_tours = sorted(
#     tours, key=lambda t: predict_score(t, user_locations, model, scaler), reverse=True
# )

# In kết quả
# for i, tour in enumerate(sorted_tours, 1):
#     score = predict_score(tour, user_locations, model, scaler)
#     print(
#         f"{i}. Locations: {tour['locations']} | Rating: {tour['rating']} | Score: {score:.4f}"
#     )

# Test với dữ liệu mới
# temp = predict_score(
#     {"locations": ["10", "14", "58"], "rating": 5}, ["58", "62"], model, scaler
# )
# print(temp)

# def upload_model_to_firebase(model, scaler):
#     init_firebase()
#     bucket = storage.bucket()

#     # Upload model
#     blob = bucket.blob(model)
#     blob.upload_from_filename(model)

#     # Upload scaler
#     blob = bucket.blob(scaler)
#     blob.upload_from_filename(scaler)

#     blob.make_public()
#     print(f" Model uploaded to: {blob.public_url}")

# def download_model_from_firebase(file_path_model, file_path_scaler):
#     init_firebase()
#     bucket = storage.bucket()
    
#     file_name_model = os.path.basename(file_path_model)
#     file_name_scaler = os.path.basename(file_path_scaler)
#     blob_model = bucket.blob(f"models/{file_name_model}")
#     blob_scaler = bucket.blob(f"models/{file_name_scaler}")

#     if blob_model.exists() and blob_scaler.exists():
#         blob_model.download_to_filename(file_path_model)
#         blob_scaler.download_to_filename(file_path_scaler)
#         return True
#     return False
    # else:
    #     model, scaler, X, y = train_model(tours, user_locations)
    #     # Lưu mô hình và scaler
    #     os.makedirs("models", exist_ok=True)
    #     joblib.dump(model, "./models/tour_model.pkl")
    #     joblib.dump(scaler, "./models/tour_scaler.pkl")
    #     return model, scaler, X, y
    


# def get_model_and_scaler():
#     model_path = "./models/tour_model.pkl"
#     scaler_path = "./models/tour_scaler.pkl"
#     storage_exists = download_model_from_firebase(model_path, scaler_path)
#     # Đã dowload về
#     if storage_exists:
#         print("Model and scaler downloaded successfully.")
#         model, scaler = load_model_from_local(model_path, scaler_path)
#         return model, scaler
#     elif storage_exists is False:
#         print("Model and scaler not found in Firebase.")
#         model, scaler = load_model_from_local(model_path, scaler_path)
#         return model, scaler
# model, scaler = get_model_and_scaler()
