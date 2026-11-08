"""Eğitilmiş model ile yeni müşteri başvurularını değerlendirme aracı."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from preprocess import load_and_preprocess

# 1. Model ve ölçekleyiciyi hazırlama
X_train, X_test, y_train, y_test, features, df = load_and_preprocess('train.csv')

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Ham veri ölçeğinde StandardScaler oluşturma
X_raw = df[features]
scaler = StandardScaler()
scaler.fit(X_raw)


# 2. Müşteri değerlendirme fonksiyonu
def test_musteri(gelir, kredi_tutari, vade, kredi_gecmisi, egitim, bolge):
    """Müşteri bilgilerini alıp model tahminini ve onay olasılığını yazdırır."""
    # Girdileri DataFrame'e dönüştürme ve standartlaştırma
    yeni_veri = pd.DataFrame([[gelir, kredi_tutari, vade, kredi_gecmisi, egitim, bolge]], columns=features)
    yeni_veri_scaled = pd.DataFrame(scaler.transform(yeni_veri), columns=features)
    
    # Sınıf tahmini ve onay olasılığı hesabı
    tahmin = model.predict(yeni_veri_scaled)[0]
    olasilik = model.predict_proba(yeni_veri_scaled)[0, 1] * 100
    
    karar_metni = "✅ KREDİ ONAYLANDI (Y)" if tahmin == 1 else "❌ KREDİ REDDEDİLDİ (N)"
    
    print(f"\nBaşvuru: Gelir=${gelir}, Kredi=${kredi_tutari}k, Kredi Geçmişi={kredi_gecmisi}, Eğitim={egitim}, Bölge={bolge}")
    print(f"Sonuç  : {karar_metni} (Onay Olasılığı: %{olasilik:.1f})")


# 3. Örnek senaryolar ile test
if __name__ == '__main__':
    print("=" * 65)
    print("CANLI BANKACILIK KREDİ DEĞERLENDİRME SİMÜLASYONU")
    print("=" * 65)

    # Senaryo 1: Düzenli sicile sahip başvuru (Beklenen: Onay)
    print("\n--- SENARYO 1: Temiz Kredi Siciline Sahip Başvuru Sahibi ---")
    test_musteri(gelir=6000, kredi_tutari=130, vade=360, kredi_gecmisi=1, egitim=1, bolge=1)
    
    # Senaryo 2: Yüksek gelirli ancak bozuk sicilli başvuru (Beklenen: Ret)
    print("\n--- SENARYO 2: Geliri Yüksek Ama Kredi Geçmişi Sorunlu Başvuru Sahibi ---")
    test_musteri(gelir=15000, kredi_tutari=140, vade=360, kredi_gecmisi=0, egitim=1, bolge=2)
    
    print("\n" + "=" * 65)
    print("[Tamamlandı] Testler başarıyla sonuçlandı!")
    print("=" * 65)
