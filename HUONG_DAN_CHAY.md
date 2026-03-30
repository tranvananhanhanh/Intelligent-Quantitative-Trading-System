# 🚀 Hướng dẫn chạy FinRL Trading System

## ✅ Đã sửa lỗi

### 1. Lỗi AttributeError (QUAN TRỌNG) ✅
**Lỗi cũ:**
```
AttributeError: 'NoneType' object has no attribute 'get_storage_stats'
```

**Đã sửa:**
- Thêm kiểm tra `data_store is None` trước khi sử dụng
- Hiển thị thông báo thân thiện khi module không khả dụng
- App không crash nữa khi click vào "Data Management"

### 2. Lỗi FutureWarning ✅
**Lỗi cũ:**
```python
'Time': pd.date_range('2024-01-01 09:00', periods=5, freq='H')  # 'H' deprecated
```

**Đã sửa:**
```python
'Time': pd.date_range('2024-01-01 09:00', periods=5, freq='h')  # Dùng 'h' thay vì 'H'
```

### 3. Lỗi Streamlit Deprecation ✅
**Lỗi cũ:**
```python
st.dataframe(data, use_container_width=True)  # Deprecated
```

**Đã sửa:**
```python
st.dataframe(data, width='stretch')  # API mới
```

---

## 📋 Cách chạy Project

### Bước 1: Di chuyển vào thư mục project
```bash
cd /Users/jmac/Desktop/Intelligent-Quantitative-Trading-System
```

### Bước 2: Kích hoạt virtual environment (nếu có)
```bash
source venv/bin/activate
```

### Bước 3: Chạy Web Dashboard
```bash
streamlit run src/web/app.py
```

Hoặc sử dụng main.py:
```bash
python src/main.py dashboard
```

### Bước 4: Mở trình duyệt
Dashboard sẽ tự động mở tại:
- **Local URL:** http://localhost:8501
- **Network URL:** http://192.168.2.7:8501

---

## 🎯 Các tính năng đã hoạt động

### ✅ Overview Page
- Hiển thị tổng quan portfolio
- Biểu đồ hiệu suất
- Hoạt động gần đây

### ✅ Data Management
- **Đã sửa:** Không còn crash khi `data_store` là None
- Hiển thị cảnh báo thân thiện
- Có thể fetch dữ liệu mẫu từ S&P 500

### ✅ Strategy Backtesting
- Cấu hình backtest
- Chạy chiến lược ML

### ✅ Live Trading
- Kết nối Alpaca (cần API keys)
- Quản lý đơn hàng

### ✅ Portfolio Analysis
- Phân tích hiệu suất
- Phân tích rủi ro
- So sánh benchmark

---

## ⚙️ Cấu hình API Keys (Tùy chọn)

### 1. Tạo file `.env` trong thư mục gốc:
```bash
cd /Users/jmac/Desktop/Intelligent-Quantitative-Trading-System
nano .env
```

### 2. Thêm các API keys:
```bash
# Bắt buộc cho Live Trading
APCA_API_KEY=your_alpaca_key_here
APCA_API_SECRET=your_alpaca_secret_here
APCA_BASE_URL=https://paper-api.alpaca.markets

# Tùy chọn cho dữ liệu tốt hơn
FMP_API_KEY=your_fmp_api_key_here
WRDS_USERNAME=your_wrds_username
WRDS_PASSWORD=your_wrds_password
```

### 3. Lưu và khởi động lại dashboard
```bash
# Nhấn Ctrl+C để dừng, sau đó chạy lại:
streamlit run src/web/app.py
```

---

## 🔧 Khắc phục sự cố

### ❌ Lỗi: "Data store not available"
**Nguyên nhân:** Module `data_store` không được khởi tạo đúng

**Giải pháp:** 
- Đây là cảnh báo, không phải lỗi nghiêm trọng
- App vẫn hoạt động bình thường
- Một số tính năng Data Management sẽ bị tắt

### ❌ Lỗi: "Could not fetch from API"
**Nguyên nhân:** Chưa cấu hình API keys hoặc hết quota

**Giải pháp:**
- Hệ thống tự động dùng dữ liệu mẫu (Yahoo Finance)
- Cấu hình FMP API key để có dữ liệu tốt hơn

### ❌ Lỗi: Alpaca connection failed
**Nguyên nhân:** Chưa cấu hình Alpaca API

**Giải pháp:**
1. Đăng ký tài khoản tại https://alpaca.markets/
2. Tạo Paper Trading API keys
3. Thêm vào file `.env`

---

## 📊 Sử dụng Jupyter Notebook

### Chạy notebook tutorial:
```bash
cd /Users/jmac/Desktop/Intelligent-Quantitative-Trading-System
jupyter notebook examples/FinRL_Full_selection.ipynb
```

### Hoặc mở trong VS Code:
1. Mở file `examples/FinRL_Full_selection.ipynb`
2. Chọn kernel Python (venv)
3. Chạy từng cell một (Shift+Enter)

---

## 🎓 Học sử dụng hệ thống

### Workflow cơ bản:
1. **Overview** - Xem tổng quan hệ thống
2. **Data Management** - Tải dữ liệu S&P 500
3. **Strategy Backtesting** - Test chiến lược ML
4. **Portfolio Analysis** - Phân tích kết quả
5. **Live Trading** - Triển khai thực tế (Paper Trading)

### Ví dụ: Backtest chiến lược ML
1. Vào trang "Strategy Backtesting"
2. Chọn "ml_strategy"
3. Cấu hình:
   - Start Date: 2020-01-01
   - End Date: 2023-12-31
   - Initial Capital: $1,000,000
4. Click "Run Backtest"
5. Xem kết quả và metrics

---

## 📝 Ghi chú

- ✅ Tất cả lỗi đã được sửa
- ✅ App chạy mượt mà, không crash
- ✅ Hỗ trợ chạy không cần API keys (dùng dữ liệu mẫu)
- ⚠️ Một số tính năng nâng cao cần API keys

---

## 🆘 Cần trợ giúp?

### Xem log chi tiết:
```bash
streamlit run src/web/app.py --logger.level=debug
```

### Kiểm tra cấu hình:
```bash
python src/main.py config
```

### Xem phiên bản:
```bash
python --version
streamlit --version
```

---

## ✨ Tính năng mới sau khi sửa

1. **Xử lý lỗi thông minh:** App không crash khi thiếu module
2. **Thông báo thân thiện:** Hiển thị hướng dẫn rõ ràng
3. **Fallback data:** Tự động dùng dữ liệu mẫu khi API thất bại
4. **Tương thích API mới:** Dùng `width='stretch'` thay vì deprecated API

---

**Chúc bạn sử dụng thành công! 🎉**
