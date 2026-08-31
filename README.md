# team__task


Bu layihə **PaySim** sintetik mobil pul əməliyyatları datasetindən istifadə edərək fraud (saxta əməliyyat) aşkarlanması üçün end-to-end feature engineering, **Feast** feature store pipeline-ı və ML modeli qurur. Bütün iş `paysim_final.ipynb` notebook-unda addım-addım sənədləşdirilmişdir.

## Problemin təsviri

PaySim kimi mobil pul əməliyyatlarında əsas problem saxta (fraudulent) əməliyyatları vaxtında aşkar etməkdir. Bunun üçün model müştərinin əvvəlki əməliyyat davranışından hesablanmış xüsusiyyətlərdən (feature) istifadə edir.

## Dataset

- **Sətir sayı:** 6,362,620
- **Sütun sayı:** 11
- **Transaction tipləri:** `PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEBIT`, `CASH_IN`
- **Fraud sayı:** 8,213 / **Non-fraud:** 6,354,407
- **Ümumi fraud rate:** 0.1291% (ciddi class imbalance)
- Ən yüksək fraud rate: `TRANSFER` (0.7688%) və `CASH_OUT` (0.1840%) əməliyyatlarında
- `step` sütunu 1 saatlıq zaman vahididir; dataset ~31 günü (1–743 step) əhatə edir
- Duplicate və missing value yoxdur

## Layihənin mərhələləri

1. **Problem Definition** — problemin və sütunların izahı
2. **Data Exploration** — dtype, missing/duplicate yoxlanışı, fraud statistikası
3. **Customer Subset Selection** — bütün fraud müştərilər + təsadüfi seçilmiş (≥3 əməliyyatlı) non-fraud müştərilərin tam tarixçəsi ilə 200,000 sətirlik idarəolunan subset
4. **Event Timestamp Creation** — `step` sütunundan (`2023-01-01` başlanğıc tarixi ilə) `event_timestamp` yaradılması
5. **Feature Engineering** — 18 yeni feature: son 24 saat / 7 gün / 14 gün üzrə transaction count, sum, mean; balans dəyişiklikləri; `time_since_last_transaction`; `log_amount`; `is_transfer`, `is_cash_out` və s.
6. **Point-in-Time Correctness** — feature-lərin yalnız keçmiş əməliyyatlardan hesablandığının, gələcək məlumatın istifadə edilmədiyinin təsdiqlənməsi
7. **Feature Validation** — 18 feature-in mövcudluğu, tip, missing/infinite dəyər yoxlanışı
8. **Parquet Offline Source** — `nameOrig`, `event_timestamp` və 18 feature `fraud_features.parquet` faylına yazılır (200,000 × 20)
9. **Feast Feature Store Setup** — `feature_store.yaml`: fayl əsaslı offline store, SQLite online store, ayrıca `registry.db`
10. **Entity Definition** — `nameOrig` join key olmaqla `customer` entity-si
11. **FeatureView Definition** — `customer_transaction_features` FeatureView-u, 18 feature, Parquet `FileSource`, 14 günlük TTL
12. **Feast Apply** — `feast apply` icrası; entity və FeatureView-nun registry-də yaradılmasının təsdiqi
13. **Historical Feature Retrieval** — `get_historical_features()` ilə point-in-time training dataset (199,953 × 21)
14. **Point-in-Time** — future information istifadəsinin (0 sətr) yoxlanılması
15. **Customer-Level Train/Val/Test Split** — `GroupShuffleSplit` ilə customer-level bölgü: Train 140,059 / Validation 29,939 / Test 29,955
16. **Data Leakage Check** — customer overlap = 0, duplicate = 0, target leakage yoxdur
17. **Class Imbalance Handling** — Train 95.87% non-fraud / 4.13% fraud; resampling tətbiq edilmir, `class_weight="balanced"` istifadə olunur
18. **Fraud Model** — Random Forest, Logistic Regression və XGBoost modellərinin training-i
19. **Model Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, SHAP feature importance
20. **Online Store Materialization** — `feast materialize` ilə 3,623,053 feature record SQLite Online Store-a yazılır
21. **Online Feature Lookup** — müəyyən `nameOrig` üçün real-time feature lookup (19/19 feature uğurla qaytarılır)
22. **PaySim Artifact Analysis** — `amount_to_balance_ratio` (importance 0.581, SHAP 5.966) ən güclü feature kimi müəyyən edilir; simulator artefaktı ola biləcəyi müzakirə olunur
23. **Artifact Ablation Analysis** — artefakt feature-ləri çıxarıldıqdan sonra: **96.88% Accuracy, 75.38% Precision, 29.62% Recall, 42.53% F1, 92.49% ROC-AUC**
24. **Conclusion** — nəticələrin ümumiləşdirilməsi

## Texnologiyalar

- **Python**, `pandas`, `numpy`
- **Feast** — feature store (offline: Parquet, online: SQLite)
- **scikit-learn** — `RandomForestClassifier`, `LogisticRegression`, `GroupShuffleSplit`, metrics
- **XGBoost** — `XGBClassifier`
- **SHAP** — feature importance / model interpretability
- **Matplotlib** — vizuallaşdırma

## Əsas nəticələr

| Metrika | Dəyər |
|---|---|
| Fraud rate (tam dataset) | 0.1291% |
| Ən vacib feature (SHAP) | `amount_to_balance_ratio` |
| Online store record sayı | 3,623,053 |
| Model (artefaktsız) Accuracy | 96.88% |
| Model (artefaktsız) Precision | 75.38% |
| Model (artefaktsız) Recall | 29.62% |
| Model (artefaktsız) F1-score | 42.53% |
| Model (artefaktsız) ROC-AUC | 92.49% |

## Fayl strukturu

```
.
├── paysim_final.ipynb        # Əsas notebook (bütün 24 mərhələ)
├── feature_store.yaml        # Feast konfiqurasiyası
├── feature_store.py          # Entity / FeatureView tərifləri
├── data/
│   └── fraud_features.parquet  # Offline feature source
└── README.md
```

## İşə salınması

```bash
pip install feast pandas scikit-learn xgboost shap matplotlib

# Feast repo-ya keç və tətbiq et
cd fraud_feature_store
feast apply

# Feature-ləri online store-a materialize et
feast materialize 2023-01-01T00:00:00 2023-01-31T23:59:59

# Notebook-u işə sal
jupyter notebook paysim_final.ipynb
```

## Qeyd

`amount_to_balance_ratio`, `balance_error` və `amount_to_newbalance_ratio` kimi feature-lər PaySim simulyatorunun daxili artefaktları ola bilər — real dünya sistemində bu feature-lərin ötürücülüyü (generalizability) əlavə yoxlanılmalıdır (bax: bölmə 22–23, Artifact Ablation Analysis).
