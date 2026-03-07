# 📊 Portfolio Analysis Guide

## ✅ Tính năng đã được thêm

### 1. **Stock Selection Sidebar**
- 📅 Date range picker (start/end date)
- 📈 Multi-select dropdown (3-10 stocks)
- 🔄 "Fetch Data" button

### 2. **Auto Fallback Yahoo Finance**
- FMP API fails (403) → **Tự động chuyển sang Yahoo Finance**
- Hiển thị data source: "FMP" hoặc "Yahoo Finance"
- Không còn crash khi FMP lỗi!

### 3. **4 Tabs phân tích**

#### Tab 1: Performance 📈
- **Normalized Price Chart** - So sánh performance (base 100)
- **Total Returns Table** - Xếp hạng theo % return
- **Price Statistics** - Start/End/Min/Max prices

#### Tab 2: Risk Analysis ⚠️
- **Risk Metrics Table**:
  - Annual Volatility
  - Maximum Drawdown
  - VaR (95%)
  - CVaR (95%)
- **Drawdown Chart** - Theo dõi drawdown theo thời gian

#### Tab 3: Returns 📊
- **Cumulative Returns Chart** - Lợi nhuận tích lũy
- **Returns Statistics Table**:
  - Mean Daily Return
  - Standard Deviation
  - Sharpe Ratio
  - Min/Max Returns

#### Tab 4: Correlation 🎯
- **Correlation Heatmap** - Ma trận tương quan
- **Correlation Summary** - Average/Highest/Lowest correlation

---

## 🧪 Cách test

### 1. Khởi động web app
```bash
cd /Users/jmac/Desktop/Intelligent-Quantitative-Trading-System
source venv/bin/activate
streamlit run src/web/app.py
```

### 2. Trong web browser
1. Click tab **"Portfolio Analysis"**
2. Trong **sidebar**, chọn:
   - Start Date: `2024-01-01`
   - End Date: `2025-01-31`
   - Stocks: Chọn 3-5 stocks (ví dụ: AAPL, MSFT, GOOGL, NVDA, TSLA)
3. Click **"🔄 Fetch Data"**

### 3. Kết quả mong đợi

**Khi FMP lỗi 403:**
```
FMP failed: 403 Client Error: Forbidden for url: ...
Fetching from Yahoo Finance: 100%|████████| 5/5 [00:02<00:00, 1.88it/s]
✅ Fetched 125 records from Yahoo Finance
```

**Khi FMP hoạt động:**
```
✅ Fetched 125 records from FMP
```

---

## 📊 Sample Output

### Performance Tab
```
Normalized Price Performance (Base 100)
AAPL: 100 → 115.2 (+15.2%)
MSFT: 100 → 108.7 (+8.7%)
GOOGL: 100 → 112.3 (+12.3%)
NVDA: 100 → 145.8 (+45.8%)
TSLA: 100 → 92.1 (-7.9%)
```

### Risk Analysis Tab
```
Ticker  | Volatility | Max Drawdown | VaR (95%) | CVaR (95%)
--------|------------|--------------|-----------|------------
AAPL    | 24.3%      | -12.5%       | -0.0234   | -0.0312
MSFT    | 21.8%      | -9.8%        | -0.0198   | -0.0256
NVDA    | 45.2%      | -28.3%       | -0.0456   | -0.0589
```

### Returns Tab
```
Ticker  | Mean Daily | Std Dev | Sharpe Ratio
--------|------------|---------|-------------
AAPL    | 0.0012     | 0.0153  | 1.24
MSFT    | 0.0008     | 0.0137  | 0.93
NVDA    | 0.0034     | 0.0285  | 1.89
```

### Correlation Tab
```
        AAPL   MSFT   GOOGL  NVDA   TSLA
AAPL    1.00   0.72   0.68   0.54   0.31
MSFT    0.72   1.00   0.76   0.61   0.29
GOOGL   0.68   0.76   1.00   0.58   0.33
NVDA    0.54   0.61   0.58   1.00   0.42
TSLA    0.31   0.29   0.33   0.42   1.00
```

---

## 🔧 Technical Details

### Data Flow
```
User selects stocks → Click "Fetch Data"
    ↓
fetch_price_data() tries FMP
    ↓ (403 Forbidden)
fetch_price_data() falls back to Yahoo Finance
    ↓
Returns DataFrame with columns:
    ['tic', 'gvkey', 'datadate', 'prcod', 'prchd', 'prcld', 'prccd', 'adj_close', 'cshtrd']
    ↓
_show_price_analysis() processes data
    ↓
Displays 4 tabs with charts & tables
```

### Key Functions

**`show_portfolio_analysis()`**
- Tạo sidebar với stock picker
- Handle "Fetch Data" button
- Gọi `_show_price_analysis()` khi có data

**`_show_price_analysis(price_data, tickers, start_date, end_date)`**
- Tab 1: Performance charts
- Tab 2: Risk metrics & drawdown
- Tab 3: Returns & Sharpe ratio
- Tab 4: Correlation matrix

**`_calculate_max_drawdown(prices)`**
- Tính maximum drawdown từ price series
- Return: float (negative number, e.g., -0.283 = -28.3%)

---

## 🎯 Next Steps

Có thể thêm:
1. **Portfolio Optimization**
   - Mean-variance optimization
   - Sharpe ratio maximization
   - Minimum volatility portfolio

2. **Benchmark Comparison**
   - So sánh với SPY, QQQ
   - Alpha/Beta calculation

3. **Export Data**
   - Download CSV
   - Export charts as PNG

4. **Advanced Analytics**
   - Rolling Sharpe ratio
   - Factor exposure analysis
   - Style drift detection

---

## 📝 Notes

- **Yahoo Finance Free Tier**: Không giới hạn price data
- **FMP Free Tier**: Bị giới hạn historical price (403 error)
- **Database Caching**: Tất cả data được cache trong SQLite
- **Performance**: Fetch 5 stocks x 1 year ≈ 2-3 seconds

---

## 🐛 Troubleshooting

### Lỗi: "No data available"
**Nguyên nhân**: Cả FMP và Yahoo Finance đều fail

**Giải pháp**:
1. Kiểm tra internet connection
2. Thử lại với date range nhỏ hơn
3. Chọn stocks khác (tránh delisted stocks)

### Lỗi: "All arrays must be of same length"
**Đã fix!** Nếu vẫn gặp, báo ngay.

### Lỗi: "KeyError: 'adj_close'"
**Đã fix!** Yahoo Finance data giờ có đủ columns.

---

## ✅ Summary

| Feature | Status |
|---------|--------|
| Stock Selection UI | ✅ |
| Date Range Picker | ✅ |
| FMP → Yahoo Fallback | ✅ |
| Performance Charts | ✅ |
| Risk Analysis | ✅ |
| Returns Analysis | ✅ |
| Correlation Matrix | ✅ |
| Error Handling | ✅ |
| Database Caching | ✅ |

**Tất cả đã hoạt động! 🎉**
