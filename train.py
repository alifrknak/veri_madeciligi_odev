"""Lojistik Regresyon model eğitimi ve parametre etki analizi."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

from preprocess import load_and_preprocess


def main():
    print("=" * 70)
    print("LOJİSTİK REGRESYON İLE KREDİ ONAY SINIFLANDIRMASI")
    print("=" * 70)

    # 1. Veri yükleme ve ön işleme
    X_train, X_test, y_train, y_test, features, df_raw = load_and_preprocess()

    # 2. Modeli eğitme
    print("[Model] Lojistik Regresyon modeli eğitiliyor...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # 3. Test verisi üzerinde tahmin ve olasılık üretme
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Pozitif sınıf (Onay) olasılığı

    # 4. Performans metrikleri
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n--- MODEL TEST PERFORMANS METRİKLERİ ---")
    print(f"Doğruluk Oranı (Accuracy) : %{acc * 100:.2f}")
    print(f"ROC-AUC Skoru             : {auc:.4f}")

    # Karmaşıklık Matrisi (Confusion Matrix)
    print("\n--- KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX) ---")
    print("                 Tahmin: Ret (0)  Tahmin: Onay (1)")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Gerçek: Ret (0)         {cm[0, 0]:<17}{cm[0, 1]}")
    print(f"Gerçek: Onay (1)        {cm[1, 0]:<17}{cm[1, 1]}")

    # Detaylı Sınıflandırma Raporu (Precision, Recall, F1)
    print("\n--- DETAYLI SINIFLANDIRMA RAPORU ---")
    print(classification_report(y_test, y_pred, target_names=['Ret (N / 0)', 'Onay (Y / 1)']))

    # 5. Parametre Etki Analizi (Katsayılar ve Odds Ratio)
    coefs = model.coef_[0]
    odds_ratios = np.exp(coefs)  # e^Beta: Birim artışın onay ihtimaline kat etkisi

    etki_df = pd.DataFrame({
        'Parametre': features,
        'Katsayı (Beta)': coefs,
        'Odds Ratio (Kat Etkisi)': odds_ratios
    }).sort_values(by='Katsayı (Beta)', ascending=False).reset_index(drop=True)

    print("=" * 70)
    print("PARAMETRE ETKİ ANALİZİ (SONUCA ETKİ EDEN FAKTÖRLER)")
    print("=" * 70)
    for idx, row in etki_df.iterrows():
        katsayi = row['Katsayı (Beta)']
        # Katsayısı sıfıra çok yakın değişkenler pratik olarak etkisizdir
        if abs(katsayi) < 0.01:
            yon = "NÖTR / ETKİSİZ (Onay Kararına Neredeyse Sıfır Etki)"
        elif katsayi > 0:
            yon = "POZİTİF (Onay Şansını Artırır)"
        else:
            yon = "NEGATİF (Reddetme İhtimalini Artırır)"
            
        print(f"{idx+1}. {row['Parametre']:<20} -> Katsayı: {katsayi:+.4f} | Odds: {row['Odds Ratio (Kat Etkisi)']:6.3f}x [{yon}]")

    print("\n" + "=" * 70)
    print("[Tamamlandı] Model eğitimi ve parametre analizi başarıyla bitti!")
    print("=" * 70)


if __name__ == '__main__':
    main()
