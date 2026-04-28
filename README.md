# Diabetes — EDA ve makine öğrenmesi (sunum projesi)

Pima Indians Diabetes veri seti ile **diyabet (Outcome) ikili sınıflandırması**: keşif grafikleri (Flutter), eğitim hattı (Python), test metrikleri ve kayıtlı model.

## Hızlı başlangıç

```bash
# 1) Python bağımlılıkları
pip install -r requirements.txt

# 2) EDA PNG’leri (Flutter “Grafikler” sekmesi)
python export_charts_for_flutter.py

# 3) Modelleri eğit, metrikleri ve ROC’u üret (Flutter “ML sonuçları” + joblib)
python src/diabetes_adaboost/training.py
# Daha uzun: AdaBoost hiperparametre araması (5 katlı CV)
python src/diabetes_adaboost/training.py --full
```

Kayıtlı modelle hızlı kontrol (konsol kodlaması için çıktı İngilizce):

```bash
python predict.py
```

## Flutter uygulaması

```bash
cd flutter_app
flutter pub get
flutter run
```

- **Grafikler:** `eda` modülündeki tüm keşif grafikleri + test ROC eğrisi.
- **ML sonuçları:** `training.py` çıktısı `assets/metrics.json` (doğruluk, dengeli doğruluk, ROC-AUC, F1, karışıklık matrisi özeti, sınıflandırma raporu).

Veri veya kod değiştiyse `export_charts_for_flutter.py` ve `src/diabetes_adaboost/training.py` komutlarını yeniden çalıştırın.

## Çıktı dosyaları

| Dosya | Açıklama |
|--------|-----------|
| `artifacts/diabetes_best_model.joblib` | Test ROC-AUC’ye göre seçilen en iyi model + ölçekleyici + medyanlar |
| `flutter_app/assets/metrics.json` | Sunum / Flutter için özet metrikler |
| `flutter_app/assets/charts/*.png` | EDA ve ROC görselleri |

## Yöntem özeti

- **Bölünme:** `train_test_split`, `test_size=0.2`, `random_state=15`.
- **Ön işleme:** Belirli sütunlarda 0 → NaN; eksikler **yalnız eğitim medyanı** ile doldurulur; `StandardScaler` (eğitim istatistikleri).
- **Modeller (hızlı mod):** Lojistik regresyon, AdaBoost, rastgele orman, KNN — test setinde karşılaştırılır; en iyi ROC-AUC’lu model kaydedilir.
- **Tam mod (`--full`):** AdaBoost için `RandomizedSearchCV` (5 katlı, `roc_auc` skoru).
- **Ek CV bilgisi:** Seçilen model için eğitim verisinde 5 katlı `roc_auc` (`cross_val_score`).

## Paket yapısı

- `src/diabetes_adaboost/` — veri, ön işleme, EDA, ek modeller (`models.py`), `training.py`, `inference.py`
- `src/diabetes_adaboost/training.py` — uçtan uca eğitim giriş noktası
- `export_charts_for_flutter.py` — EDA PNG üretimi
