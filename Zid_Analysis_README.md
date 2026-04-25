# 🛒 Business Performance Analysis — Zid Platform

> تحليل شامل لأداء المتاجر على منصة **زد** للتجارة الإلكترونية باستخدام Python وأدوات تحليل البيانات.

---

## 📌 Project Overview

تحليل بيانات حقيقية لـ **3,851 متجر** على منصة زد، يهدف إلى فهم أداء المتاجر وطرق الشحن وسلوك العملاء وتقديم توصيات مبنية على البيانات.

**📅 Date:** October 2024  
**👤 Author:** Mousa Alubaid  
**🏢 Platform:** Zid (زد) — Saudi E-commerce Platform

---

## 🎯 Tasks Covered

| # | Task | Description |
|---|------|-------------|
| 1 | **KPI Analysis** | قياس مؤشرات الأداء الرئيسية لكل متجر |
| 2 | **Shipping Analysis** | تحليل طرق الشحن ومعدلات التأخير والإلغاء |
| 3 | **Customer Segmentation** | تقسيم العملاء باستخدام RFM Analysis |
| 4 | **Challenges & Recommendations** | تحديد التحديات وتقديم توصيات قابلة للتطبيق |

---

## 📊 Key Findings

### 🚚 Shipping Analysis
- **Aramex** = أسرع شركة توصيل من حيث متوسط وقت التسليم
- شركات مثل **توصيل مرند** تمتلك أعلى معدل إلغاء (58.8%)
- **ZidShip** تحتل المرتبة الخامسة ولكن تعاني من معدل إلغاء مرتفع

### 👥 Customer Segmentation (RFM)
```
Casual Customers   → 109,954  (الأكبر)
New Customers      →  69,087
Potential          →  43,281
Elite              →   8,658
At-Risk            →   5,570
Loyal              →   1,913
```

### 🏪 Top Industries by Revenue (SAR)
1. 🥇 Fragrances — 16M SAR
2. 🥈 Fashion & Jewelry — 13M SAR
3. 🥉 Foods — 12.7M SAR

### 🗺️ Top Cities by Orders
- **Riyadh** → 70,187 orders (المرتبة الأولى)
- **Jeddah** → 23,361
- **Dammam** → 14,143

---

## 🛠️ Tools & Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge)

- **Python** — Data processing & analysis
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Data visualization
- **RFM Analysis** — Customer segmentation methodology
- **Excel** — KPI reporting per store

---

## 📁 Project Structure

```
zid-analysis/
│
├── 📊 Report_Business_Performance_Analyst.pdf   ← Full analysis report
├── 📋 KPI_Stores.xlsx                           ← KPIs for all 3,851 stores
├── 📦 Shipping_Analysis.xlsx                    ← Shipping performance data
├── 👥 Customer_Segmentation.xlsx                ← RFM segmentation results
└── 📓 Analysis.ipynb                            ← Jupyter Notebook
```

---

## 💡 Key Recommendations

1. **تحسين تجربة الدفع** — تقليل الإلغاءات عبر تسهيل خيارات الدفع
2. **الشراكة مع Aramex** — للمتاجر ذات معدلات الإلغاء المرتفعة
3. **تطوير ZidShip** — تحسين الموثوقية لرفع تنافسيتها
4. **استهداف العملاء النشطين** — ببرامج ولاء مخصصة لكل شريحة
5. **تنظيف البيانات** — معالجة أخطاء الإدخال (أوقات توصيل سالبة / 300 يوم)

---

## 📈 Visualizations Included

- ✅ Top 10 Shipping Methods (Usage Count)
- ✅ Average Delivery Time per Carrier
- ✅ Cancellation Rate by Shipping Method
- ✅ Customer Segmentation Counts (RFM)
- ✅ Top 10 Industries by Orders & Revenue
- ✅ Top 10 Cities by Shipment Volume
- ✅ Payment Status vs Cancellation Rate

---

*Built with 💙 | Data-Driven Decisions for Saudi E-commerce*
