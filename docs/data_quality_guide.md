# Data Quality Assessment Guide

## Phần Data Quality là gì?

**Data Quality (Chất lượng Dữ liệu)** là tính năng tự động **chấm điểm chất lượng** của dữ liệu được fetch/process từ Yahoo Finance.

Nó đánh giá dữ liệu từ 4 khía cạnh:

## 4 Tiêu chí Chấm Điểm

### 1. **Completeness (Tính Đầy Đủ)** - 0-100%
- **Tính toán**: Tỷ lệ % giá trị không phải NULL
- **Ý nghĩa**: 
  - 100% = Tất cả giá trị đều có dữ liệu
  - 80-100% = Rất đầy đủ
  - 70-80% = Đầy đủ
  - <50% = Thiếu đáng kể
- **Ví dụ**:
  ```
  Cột revenue: 250/250 giá trị = 100%
  Cột total_assets: 175/250 giá trị = 70%
  Completeness = (100 + 70 + ...) / 13 fields = 71.43%
  ```

### 2. **Accuracy (Tính Chính Xác)** - 0-100%
- **Tính toán**: Kiểm tra giá trị hợp lệ + outlier detection
- **Kiểm tra**:
  - **Fundamental**: PE > 0, Revenue > 0, không outlier quá 3 IQR
  - **Price**: High ≥ Low, Open/Close trong [Low, High], Volume ≥ 0
- **Ví dụ**:
  ```
  Price checks:
  - Low ≤ High: ✓ 100%
  - Open in [Low, High]: ✓ 100%
  - Volume ≥ 0: ✓ 100%
  Accuracy = 100%
  ```

### 3. **Consistency (Tính Nhất Quán)** - 0-100%
- **Tính toán**: Kiểm tra format và data type consistency
- **Kiểm tra**:
  - Format ngày tháng hợp lệ (YYYY-MM-DD)
  - Các cột número không có text
  - Không có hàng trùng lặp
- **Ví dụ**:
  ```
  Date format valid: ✓ 100%
  Numeric columns all numbers: ✓ 100%
  No duplicates: ✓ 100%
  Consistency = 100%
  ```

### 4. **Timeliness (Tính Kịp Thời)** - 0-100%
- **Tính toán**: Làm sao dữ liệu so với hôm nay?
- **Điểm dựa trên ngày dữ liệu mới nhất**:
  - 100% = ≤ 1 ngày (cập nhật hôm nay/hôm qua)
  - 90%  = ≤ 1 tuần
  - 70%  = ≤ 1 tháng
  - 50%  = ≤ 3 tháng
  - 20%  = > 3 tháng (cũ)
- **Ví dụ**:
  ```
  Ngày dữ liệu mới nhất: 2023-12-31
  Hôm nay: 2026-03-16 (cách > 3 tháng)
  Timeliness = 20%
  ```

### 5. **Overall Score (Điểm Tổng Thể)** - 0-100%
- **Tính toán**: Trung bình của 4 tiêu chí
- **Formula**: (Completeness + Accuracy + Consistency + Timeliness) / 4

## Biểu Tượng Trạng Thái

| Điểm | Emoji | Trạng Thái | Chất Lượng |
|------|-------|-----------|-----------|
| 95-100 | 🟢 | Excellent | Tuyệt vời - Sử dụng ngay được |
| 80-94 | 🟡 | Good | Tốt - Chấp nhận được |
| 60-79 | 🟠 | Fair | Trung bình - Cần chú ý |
| <60 | 🔴 | Poor | Kém - Cần xử lý thêm |

## Ví Dụ Thực Tế

### Fundamental Data từ Yahoo Finance
```
Completeness: 71.43%  
  → Vì Yahoo không cung cấp total_assets, total_liabilities

Accuracy: 89.74%      
  → Vì PE, Revenue mostly valid, ít outliers

Consistency: 100.00%  
  → Format ngày tháng chuẩn, không trùng lặp

Timeliness: 20.00%    
  → Dữ liệu cũ (>3 tháng từ fetch date)

Overall: 70.29% (🟠 Fair)
```

### Price Data từ Yahoo Finance
```
Completeness: 100.00%  
  → Tất cả open, high, low, close, volume có đủ

Accuracy: 100.00%     
  → OHLC relationships hoàn hảo

Consistency: 100.00%  
  → Format đồng nhất

Timeliness: 20.00%    
  → Dữ liệu cũ (>3 tháng)

Overall: 80.00% (🟡 Good)
```

## Khi Nào Auto-Assessment Chạy?

### 1. **After Fetch Fundamentals**
```
Button: "Fetch Fundamental Data"
→ Auto-display Data Quality score ngay dưới nút
→ Người dùng biết quality của data vừa fetch
```

### 2. **After Process Raw Data**
```
Button: "Process Raw Data"
→ Auto-assess cả Fundamentals + Prices sau processing
→ Hiển thị side-by-side comparison
```

### 3. **Manual Check**
```
Tab: "Data Quality"
Button: "🔍 Assess Data Quality"
→ Cho phép riêng check quality bất kỳ lúc nào
```

## Cách Cải Thiện Data Quality

| Vấn Đề | Nguyên Nhân | Giải Pháp |
|--------|-----------|----------|
| Completeness thấp | Yahoo không cung cấp field | Chỉ sử dụng available fields |
| Accuracy thấp | Outliers trong data | Detection automatic, không affect training |
| Consistency thấp | Date format lỗi | Validate trước process |
| Timeliness thấp | Dữ liệu cũ | Fetch data mới/gần đây |

## Lưu Ý Khi Sử Dụng

✅ **Data Quality → 80%+**: Sử dụng trực tiếp cho model training
✅ **Data Quality → 60-80%**: Chấp nhận được, có thể dùng
❌ **Data Quality < 60%**: Cần investigate, không nên train model

---

## Tổng Kết

- **Phần Data Quality** = Tự động chấm điểm data sau mỗi lần fetch/process
- **Auto-Assessment** = Chạy automatically sau Fetch Fundamental + Process Raw Data
- **4 Tiêu chí**: Completeness, Accuracy, Consistency, Timeliness
- **Overall Score**: Trung bình 4 tiêu chí, 0-100%
- **Biểu tượng**: 🟢 Excellent (95+), 🟡 Good (80-94), 🟠 Fair (60-79), 🔴 Poor (<60)
