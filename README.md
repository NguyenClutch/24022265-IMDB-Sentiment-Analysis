# Bài tập RNN/LSTM/GRU: Sentiment analysis trên IMDB

Tệp chính: **IMDB_LSTM_Comparison.ipynb**. Phần giải thích và phân tích được viết bằng tiếng Việt.


## Cấu trúc dự án

```text
IMDB_LSTM_Comparison.ipynb   # Bài làm hoàn chỉnh, đã có output thực nghiệm
run_experiment.py          # Chạy lại toàn bộ thí nghiệm bằng Python
requirements.txt           # Các thư viện cần thiết
results/                   # Bảng kết quả, lịch sử huấn luyện và biểu đồ
```

Để xem bài làm, mở **[notebook đã chạy](IMDB_LSTM_Comparison.ipynb)** ngay trên GitHub. Không cần chạy lại để xem kết quả.

![So sánh các mô hình](results/comparison.png)

### Chạy bằng Python

```bash
pip install -r requirements.txt
python run_experiment.py
```

Lệnh này huấn luyện lại cả bốn mô hình và tạo các tệp kết quả. Dataset được tải tự động vào `data/`, thư mục này được bỏ qua khi đưa lên GitHub.

## Nội dung đáp ứng yêu cầu

- Chọn LSTM trong nhóm RNN/LSTM/GRU.
- Phân loại tích cực/tiêu cực trên 50.000 review IMDB có nhãn.
- So sánh với mạng neural cơ bản MLP, Linear SVM và Logistic Regression.
- Train/validation/test: 20.000/5.000/25.000, dùng chung cho mọi mô hình.
- Từ vựng và IDF chỉ học trên train; checkpoint và C chọn bằng validation.
- Báo cáo accuracy, precision, recall, F1, ROC-AUC, thời gian; biểu đồ và ma trận nhầm lẫn; phân tích lỗi và hạn chế.

## Chạy trên Google Colab

1. Tải notebook lên https://colab.research.google.com/.
2. Có thể chọn Runtime → Change runtime type → GPU để chạy nhanh hơn.
3. Nếu thiếu thư viện, bỏ dấu `#` ở dòng `%pip install` trong ô đầu, chạy ô đó rồi khởi động lại kernel.
4. Chọn Runtime → Run all. Cần Internet ở lần tải dữ liệu đầu.
5. Lưu/tải notebook sau khi hoàn tất để giữ các output.

## Chạy trên máy cá nhân

Khuyến nghị Python 3.11–3.13. Trong môi trường riêng, cài thư viện bằng `pip install -r requirements.txt`, sau đó chạy `jupyter notebook` và mở notebook. Tốc độ phụ thuộc phần cứng; không có thời gian chạy cố định.

Notebook tải hai split parquet từ https://huggingface.co/datasets/stanfordnlp/imdb và kiểm tra số lượng/nhãn. Dữ liệu không nhãn không được sử dụng. Thư mục `results/` chứa các bảng, biểu đồ và thông tin môi trường của lần chạy. Cùng seed vẫn có thể cho kết quả hơi khác khi thay phần cứng hoặc phiên bản thư viện.

## Nộp bài

Có thể nộp trực tiếp tệp `.ipynb` đã chạy. Nếu nộp GitHub, tải notebook, README, requirements và thư mục results lên kho mã của bạn rồi nộp liên kết. Không đưa thư mục `data/` hoặc dữ liệu IMDB lên kho mã. Điền thông tin sinh viên theo quy định lớp nếu cần.

Nguồn bài toán: Maas et al. (2011), https://aclanthology.org/P11-1015/ và https://ai.stanford.edu/~amaas/data/sentiment/.

## Kết quả đã chạy thực tế

Notebook đã chạy toàn bộ trên CPU với seed 42. Các output đã được lưu trong tệp `.ipynb`.

| Mô hình | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.8992 | 0.8997 | 0.9632 |
| Linear SVM | 0.8964 | 0.8966 | 0.9616 |
| MLP | 0.9008 | 0.9012 | 0.9649 |
| LSTM | 0.8280 | 0.8363 | 0.9010 |

MLP có F1 cao nhất trong lần chạy. LSTM dùng tối đa 256 token đầu; khoảng 29,08% review train bị cắt. Kết quả này chỉ áp dụng cho cấu hình và một seed trong notebook, không chứng minh thứ hạng chung của các họ mô hình. Thông tin phiên bản và dấu kiểm SHA256 của dữ liệu có trong `results/run_metadata.json`.
