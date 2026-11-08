"""Kredi verisi ön işleme ve hazırlık modülü."""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Modelde kullanılacak girdi özellikleri ve hedef değişken
FEATURES = [
    'ApplicantIncome',    # Aylık gelir
    'LoanAmount',         # Kredi tutarı
    'Loan_Amount_Term',   # Kredi vadesi (ay)
    'Credit_History',     # Kredi geçmişi (1: Olumlu, 0: Olumsuz)
    'Education',          # Eğitim durumu (1: Mezun, 0: Mezun değil)
    'Property_Area'       # Bölge (0: Rural, 1: Semiurban, 2: Urban)
]
LABEL = 'Loan_Status'     # Hedef etiket (1: Onay, 0: Ret)


def load_and_preprocess(file_path='train.csv'):
    """Ham veriyi okur, temizler, kodlar, böler ve ölçekler."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HATA: '{file_path}' dosyası bulunamadı! Lütfen dosyanın proje dizininde olduğundan emin olun.")

    print(f"[1/4] Veri seti yükleniyor: {file_path}")
    df = pd.read_csv(file_path)
    df = df[FEATURES + [LABEL]].copy()

    # Eksik verileri doldurma (Imputation)
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())  # Aykırı değerlere dirençli medyan
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(360.0)          # En sık görülen vade (mod)
    df['Credit_History'] = df['Credit_History'].fillna(1.0)                # Çoğunluk sınıfı (1.0)
    print("[2/4] Eksik veriler tamamlandı (Imputation yapıldı).")

    # Kategorik değişkenleri sayısallaştırma (Encoding)
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['Property_Area'] = df['Property_Area'].map({'Rural': 0, 'Semiurban': 1, 'Urban': 2})
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    print("[3/4] Kategorik değişkenler sayısallaştırıldı (Encoding yapıldı).")

    # Girdi (X) ve hedef (y) ayrımı
    X = df[FEATURES]
    y = df[LABEL]

    # Eğitim (%80) ve test (%20) ayrımı (sınıf oranlarını koruyarak)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # NEDEN YAPILDI:
    # 1. StandardScaler: Gelir (50.000$) gibi büyük sayılar 0-1 olan kredi sicilini ezmesin,
    #    tüm değişkenler eşit şartta yarışsın ve katsayıları kıyaslanabilsin diye yapıldı.
    # 2. fit(train) / transform(test): Model test verisini önceden görüp ezberlemesin (veri sızıntısı
    #    / data leakage önlensin), gerçek hayattaki gibi sadece eğitim bilgisiyle değerlendirsin diye yapıldı.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # DataFrame formatını ve sütun isimlerini koruma
    X_train = pd.DataFrame(X_train_scaled, columns=FEATURES, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=FEATURES, index=X_test.index)

    print(f"[4/4] Veri hazırlandı! (Eğitim: {len(X_train)} satır, Test: {len(X_test)} satır)\n")
    return X_train, X_test, y_train, y_test, FEATURES, df


if __name__ == '__main__':
    # Test çalıştırması: veriyi hazırlar ve ilk 3 satırı gösterir
    X_train, X_test, y_train, y_test, features, df = load_and_preprocess()
    print("Örnek Standartlaştırılmış Eğitim Verisi (İlk 3 Satır):")
    print(X_train.head(3))
