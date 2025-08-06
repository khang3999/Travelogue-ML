from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#  Hàm training
def train_model(tours, user_locations):
    # user_location_vector = one_hot_from_cities(user_locations, city_to_index)
    X = []
    y = []
    # Vì fit() của MinMaxScaler nhận về một 2D array, nên ta cần chuyển đổi các rating thành dạng [[rating1], [rating2], ...]
    ratings = [[tour["rating"]] for tour in tours]
    scaler = MinMaxScaler()
    # Tự nội suy đưa min về 0 và max về 1 các giá trị giữa cho phù hợp
    scaler.fit(ratings)

    for tour in tours:
        tour_locations_set = set(tour["locations"])
        user_locations_set = set(user_locations)

        # Giao nhau
        overlap_count = len(tour_locations_set & user_locations_set)
        max_len = max(len(tour["locations"]), len(user_locations))
        delta = 1 - (overlap_count / max_len) if max_len > 0 else 1
        # Chuyển đổi rating về dạng đã nội suy. scaler.transform trả mảng 2 chiều [[0.75]] nên lấy 2 lần [0][0]
        rating_scaled = scaler.transform([[tour["rating"]]])[0][0]

        features = [overlap_count, delta, rating_scaled]
        X.append(features)

        # Tạo kết quả (label) cho từng dữ liệu của mô hình nếu có giao nhau và rating >= 2.5 (Dùng toán tử 3 ngôi)
        label = 1 if overlap_count > 0 and tour["rating"] >= 2.5 else 0
        y.append(label)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, scaler, X, y

# Tính toán điểm số dự đoán cho từng tour - đưa qua backend
# Dựa trên số lượng địa điểm giao nhau và rating đã nội suy
def predict_score(tour, user_locations, model, scaler):
    # Đếm tỉnh trùng
    overlap_count = len(set(tour["locations"]) & set(user_locations))
    # Tính delta - độ lệch giữa số lượng địa điểm giao nhau và tổng số địa điểm
    max_len = max(len(tour["locations"]), len(user_locations))
    delta = 1 - (overlap_count / max_len) if max_len > 0 else 1
    # Chuyển đổi rating về dạng đã nội suy
    rating_scaled = scaler.transform([[tour["rating"]]])[0][0]
    # Tạo đặc trưng (X) của từng tour để mô hình dự đoán
    features = [[overlap_count, delta, rating_scaled]]
    return model.predict_proba(features)[0][1]


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
    
    
