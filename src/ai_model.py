from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


# Dữ liệu cho training
# # {
#         "behavior": "40,38",
#         "id": "-OVkNEIIdSdXXUSOPsPG",
#         "locations": [
#             "48"
#         ],
#         "rating": 0
#     }


# Hàm xử lý dữ liệu thành feature
def prepare_data(tour_data_set):
    X = []
    y = []
    for tour in tour_data_set:
        tour_locations = tour["locations"]
        if isinstance(tour["behaviors"], str):
            behavior = [b.strip() for b in tour["behaviors"].split(",")]
        else:
            behavior = [str(b).strip() for b in tour["behaviors"]]
        rating = tour["rating"]
        # Giao nhau
        overlap_count = len(set(tour_locations) & set(behavior))
        max_len = max(len(tour_locations), len(behavior))
        delta = (overlap_count / max_len) if max_len > 0 else 0
        # # Chuyển đổi rating về dạng đã nội suy. scaler.transform trả mảng 2 chiều [[0.75]] nên lấy 2 lần [0][0]
        # rating_scaled = scaler.transform([[tour["rating"]]])[0][0]
        # Chuẩn hóa rating về [0, 1]
        rating_scaled = float(rating) / 5.0

        feature = [overlap_count, delta, rating_scaled]
        X.append(feature)

        # Tạo kết quả (label) cho từng dữ liệu của mô hình nếu có giao nhau và rating >= 2.5 (Dùng toán tử 3 ngôi)
        label = 1 if overlap_count > 0 and rating_scaled >= 0.5 else 0
        y.append(label)
        print(X)
        print(y)
    return X, y


#  Hàm training
def train_model(tour_dataset):
    # Load mô hình đã lưu, nếu không có thì tạo mới
    try:
        model = joblib.load("models/tour_model.pkl")
    except FileNotFoundError:
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_features=2)
    # Tạo đặc trưng từ dữ liệu
    X, y = prepare_data(tour_dataset)
    # Kiểm tra type của mô hình
    print("Type of model before training:", type(model))
    # Train mô hình
    model.fit(X, y)
    return model


# Dữ liệu cho đánh giá
# Tính toán điểm số dự đoán cho từng tour - đưa qua backend
# Dựa trên số lượng địa điểm giao nhau và rating đã nội suy
def predict_score(tour_dataset, model):
    # X là một mảng các feature
    X, y = prepare_data(tour_dataset)

    # result =
    # [
    #   [0.2, 0.8],  # Tour 1 → 80% phù hợp
    #   [0.6, 0.4],  # Tour 2 → 40% phù hợp
    #   [0.1, 0.9]   # Tour 3 → 90% phù hợp
    # ]
    result = model.predict_proba(X)
    return result


# Đánh giá mô hình
def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy : {acc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1 Score : {f1:.2f}")
