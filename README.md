# IRONMAN 70.3 · Dự báo thời gian về đích

Repo chứa code modeling, model đã train và kết quả cho demo hội thảo. Hai bảng Parquet đầy đủ nằm trong gói bàn giao IRONMAN_Khue_Modeling.zip của nhóm.

## Mở kết quả

- `predictions.parquet` là bảng dự báo chính. Mỗi dòng ứng với một lượt × giai đoạn × checkpoint × phương pháp. Các khóa là `resultid, split, checkpoint, model, seed`. Bản ghi dùng `resultid` để ghép Supabase. Seed 0 biểu thị công thức hoặc LR tất định. Hai cột `pred_finish_seconds` và `actual_finish_seconds` là tổng thời gian từ xuất phát đến đích, đơn vị giây.
- `metrics/all_models.csv` báo MAE, RMSE, bias và P90 sai số tuyệt đối theo phút, số lượt fit/test và số giải. `metrics/example.csv` là bản CSV nhỏ chứa 10 lượt test có thật qua cả ba mốc, phù hợp để thử luồng đọc file. Các lượt được lấy theo ID, không chọn theo sai số. Khi làm demo chỉ mở actual ở bước cuối.
- `cohort_manifest.parquet` liệt kê danh sách ID và trạng thái fit/test/unused/excluded_total của hai lần đánh giá. Các bảng `models/*_baseline.csv` lưu trung vị và cỡ từng nhóm để dùng lại B2.
- `models/I_C0.cbm` đến `models/II_C2.cbm` là sáu mô hình CatBoost đã fit. Bảng dự báo là đầu ra hữu ích nhất để so sánh ghép cặp.
- `v1/ML_model.ipynb` là bản notebook gốc có output, lấy từ Drive Paper/Model. Bên cạnh là bảng điểm lịch sử và cấu hình trích từ notebook. V1 dùng LR/RF ở C0/C2, chia theo năm 2004–2020 và 2021–2023. Mẫu, biến và tiền xử lý V1 khác lần chạy ba mốc.

## Thiết kế lần chạy ba mốc

Evaluation I fit 2004–2016, test 2019–2020 cùng nguồn Kaggle. Evaluation II fit 2004–2018, test 2021–2023 từ nguồn Endurance. Các năm dành cho calibration theo protocol được giữ ra ngoài fit. Lần chạy này chỉ đánh giá dự báo điểm, seed mô hình 42.

Bộ dữ liệu là ba bảng CSV trong thư mục cleaned_data của nhóm. Link, SHA-256 và tham số nằm trong config. Đối chiếu đủ 1.153.194 dòng từ vw_valid_c1c2 của Supabase: sau bộ lọc dân số và split dương, hai nguồn cho cùng 1.152.658 resultid, mọi cột dùng để lập mẫu, train và chấm đều khớp. Xem metrics/snapshot_comparison.json. Giữ kết quả cá nhân Age-Group, giới M/F, nhóm tuổi 18–24 đến 70–74, thời gian bơi, đạp và hoàn thành hữu hạn, dương. Cùng danh sách lượt ở cả ba mốc. Thiếu quốc gia giữ nhãn riêng, mã NA của Namibia được giữ nguyên. Không cắt phân vị FinishTime. Kiểm tra bản ghi và khóa liên kết trước khi ghép.

Khi soát tính nhất quán sau lượt chạy đầu, có 113 lượt ở 2021–2023 có FinishTime nhỏ hơn tổng SwimTime và BikeTime. File dự báo chính và all_models.csv chỉ giữ lượt có FinishTime > SwimTime + BikeTime. Không có lượt thuộc mẫu fit nào ở hai giai đoạn bị ảnh hưởng. Bảng before_consistency_check.csv lưu điểm trước bước QC để Hội kiểm tra độ nhạy. cohort.csv đếm sau lọc nhân khẩu và split dương, trước QC này. Dữ liệu đã qua xử lý của nhóm; chưa xác minh riêng cự ly từng giải hoặc alias địa điểm.

C0 có giới, cận dưới nhóm tuổi, quốc gia VĐV, địa điểm và năm. C1 mở thêm SwimTime. C2 mở thêm BikeTime. Không dùng RunTime, thứ hạng, ratio hoặc transition làm đặc trưng. F3 dùng tên địa điểm trong snapshot sau khi cắt khoảng trắng. Cả ba thuật toán học trực tiếp FinishTime.

B0 là trung vị FinishTime toàn fit. B1 là trung vị theo giới × nhóm tuổi trước giải. B2 cộng tổng các split đã biết với trung vị của phần thời gian còn lại theo giới × nhóm tuổi, lùi về giới rồi toàn fit khi dưới 30 lượt. Ở C0, B1 và B2 trùng nhau. Một giá trị `pred_finish_seconds` của B2 ở C2 đã bao gồm cả thời gian bơi và đạp.

CatBoost_ref là lần chạy tham chiếu với cấu hình cố định trong run.json. Notebook V1 đã kiểm tra chỉ dùng LR/RF. Không có bằng chứng rằng 120 vòng là cấu hình CatBoost mà nhóm từng dùng. Chưa chạy toàn bộ lưới tuning, ba seed, hiệu chỉnh khoảng hoặc kiểm định theo địa điểm. RF dùng lại 200 cây, depth 12, min_samples_leaf 50, seed 42; LR và RF lần này dùng mã hóa one-hot học từ tập fit. Các điểm LR/RF chạy lại nằm trong bảng kết quả chung.

## Chạy lại

Dùng Python 3.12. Tải ba CSV có tên chính xác trong config/sources.json vào cùng thư mục. So khớp SHA-256 nếu muốn tái lập đúng snapshot. Chạy từ thư mục chứa README.

```bash
pip install -r requirements.txt
mkdir -p predictions models metrics
python scripts/run.py /path/to/csv basic
python scripts/run.py /path/to/csv catboost
python scripts/run.py /path/to/csv rf
python scripts/manifest.py /path/to/csv
python scripts/export.py /path/to/csv
```

Đọc một phần bảng bằng pandas hoặc chuyển một giai đoạn sang CSV nén.

```python
import pandas as pd
df = pd.read_parquet("predictions.parquet", filters=[("split", "==", "II"), ("model", "==", "CatBoost_ref")])
df.to_csv("catboost_eval_II.csv.gz", index=False)
print(df.groupby("checkpoint").resultid.nunique())
```

## Mở một checkpoint từ model đã lưu

Ví dụ sau dùng đúng thứ tự feature của C1. Các giá trị đưa vào là một ví dụ định dạng. Với C0 bỏ swimtime, với C2 thêm biketime sau swimtime. Chọn file .cbm của chính giai đoạn và checkpoint cần dùng. Các model của hai giai đoạn được fit bằng các năm khác nhau.

```python
import json
import pandas as pd
from catboost import CatBoostRegressor

cfg = json.load(open("config/run.json"))
cp = "C1"
features = cfg["features_pre"] + cfg["checkpoints"][cp]
row = {"gender": "F", "ageband": 35, "countryiso": "US",
       "eventlocation": "IRONMAN 70.3 Florida", "eventyear": 2022, "swimtime": 2100}
model = CatBoostRegressor()
model.load_model(f"models/II_{cp}.cbm")
pred_finish_seconds = model.predict(pd.DataFrame([row], columns=features))[0]
```
