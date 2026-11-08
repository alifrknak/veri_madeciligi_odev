# YÜKSEK LİSANS VERİ MADENCİLİĞİ PROJE RAPORU
## Lojistik Regresyon ile Bireysel Kredi Onay Sınıflandırması ve Parametre Etki Analizi

---

**Öğrenci:** Yüksek Lisans Veri Madenciliği Dersi  
**Veri Seti Sahibi & Kaynağı:** Kaggle - *Loan Prediction Problem Dataset* (Veri Sahibi: **altruistdelhite04**)  
**Kullanılan Model:** Lojistik Regresyon (Logistic Regression)  
**Doğrulama Yöntemi:** %80 Eğitim (Train), %20 Test (Hold-out)  

---

## 1. GİRİŞ VE PROJE ÖZETİ

Bu projede, bir bankaya bireysel kredi başvurusunda bulunan müşterilerin temel finansal ve demografik bilgileri kullanılarak, başvurunun onaylanıp onaylanmayacağı (**`Loan_Status`**: `Y: Onay`, `N: Ret`) Lojistik Regresyon modeli ile sınıflandırılmıştır. 

Çalışmada kullanılan veri seti, Kaggle platformunda araştırmacı ve veri bilimci **altruistdelhite04** tarafından açık erişimli olarak paylaşılan *"Loan Prediction Problem Dataset"* [1] üzerinden elde edilmiştir. Orijinal veri setinde yer alan 12 sütun arasından karar mekanizmasında en belirleyici olan **6 parametre** seçilmiştir:

1. `Credit_History` (Kredi Geçmişi)
2. `ApplicantIncome` (Başvuran Geliri)
3. `LoanAmount` (Talep Edilen Kredi Tutarı)
4. `Loan_Amount_Term` (Kredi Vadesi)
5. `Education` (Eğitim Durumu)
6. `Property_Area` (Yaşanılan Bölge)

Modelimiz test verisi üzerinde **%85.37 Doğruluk (Accuracy)**, **%90.32 F1-Skoru** ve **0.8248 ROC-AUC** skoru elde etmiştir.

---

## 2. NEDEN LOJİSTİK REGRESYON (LOGISTIC REGRESSION) SEÇİLDİ?

Bankacılık ve finans alanındaki kredi skorlama (credit scoring) projelerinde Lojistik Regresyon en çok tercih edilen ve bu veri setine **en uygun** algoritmadır [2, 3]. Bu seçimin 4 temel akademik ve sektörel gerekçesi bulunmaktadır:

### 2.1. İkili (Binary) Sınıflandırma Mantığına Doğal Uyumu
Projedeki problem ikili (binary) bir sınıflandırmadır: Kredi ya onaylanır ($Y = 1$) ya da reddedilir ($N = 0$). Lojistik Regresyon, Sigmoid fonksiyonunu kullanarak girdi parametrelerini 0 ile 1 arasında sürekli bir **olasılık değerine ($P$)** dönüştürür:

$$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \dots + \beta_k X_k)}}$$

Bu sayede model sadece "Onay/Ret" demekle kalmaz; örneğin *"Bu müşterinin onay alma ihtimali %79.9'dur"* şeklinde finansal karar vericilere doğrudan olasılıksal güven skoru sunar.

### 2.2. Bankacılıkta Açıklanabilirlik (Explainability / Interpretability) Zorunluluğu
Finansal piyasalarda ve yasal düzenlemelerde (BDDK, Basel II/III kriterleri, KVKK/GDPR), bir müşterinin kredi başvurusu reddedildiğinde bankanın **"Neden reddedildiğini"** şeffaf bir biçimde gerekçelendirmesi yasal bir zorunluluktur.
- Derin öğrenme ağları (Neural Networks) veya karmaşık ağaç toplulukları (Random Forest, XGBoost) birer **"Kara Kutu" (Black-Box)** modeldir; kararın arkasındaki matematiksel bağı doğrudan tek bir formülle açıklayamazlar.
- **Lojistik Regresyon ise tamamen şeffaftır.** Modelin her bir parametre için bulduğu bir katsayı ($\beta$) vardır ve bu katsayılar doğrudan iş ve risk birimlerince denetlenebilir.

### 2.3. Veri Seti Boyutu ve Aşırı Öğrenme (Overfitting) Riski
Kullandığımız veri seti 614 satırdan oluşmaktadır. Bu ölçekteki veri setlerinde derin ve dallanmış ağaç algoritmaları veriyi ezberleme (overfitting) eğilimindedir. Lojistik Regresyon ise doğrusal karar sınırı ve düzenlileştirme kabiliyeti sayesinde küçük-orta veri setlerinde aşırı öğrenmeye düşmeden genelleme yapar. Nitekim test kümesinde elde edilen **%85.37 doğruluk ve 0.8248 AUC skoru**, modelin veri setini mükemmel temsil ettiğini kanıtlamaktadır.

### 2.4. Parametre Etkisinin Doğrudan Matematiksel Hesaplanabilmesi (Odds Ratio)
Lojistik Regresyonun en büyük üstünlüğü, katsayıların üstel değeri ($e^\beta$) alınarak **Odds Ratio (Risk Katı)** hesabının doğrudan yapılabilmesidir. Böylece *"Kredi geçmişi temiz olan bir adayın onay alma şansı, geçmişi bozuk olan adaya göre kaç kat fazladır?"* sorusu tam bir kesinlikle hesaplanabilmektedir.

---

## 3. VERİ ÖN İŞLEME VE TEMİZLEME ADIMLARI (`preprocess.py`)

1. **Eksik Verilerin Doldurulması (Imputation):**
   - `LoanAmount` (Kredi Tutarı): Uç değerlerden etkilenmemek adına **medyan (128.0)** ile dolduruldu.
   - `Loan_Amount_Term` (Vade): En yaygın standart konut kredisi vadesi olan **mod (360 ay)** ile dolduruldu.
   - `Credit_History` (Kredi Geçmişi): Veri setinde en çok görülen sınıf olan **mod (1.0 - Temiz Sicil)** ile tamamlandı.
2. **Kategorik Değişkenlerin Kodlanması (Encoding):**
   - `Education`: Lisans/Lisansüstü (`Graduate = 1`), Mezun Değil (`Not Graduate = 0`).
   - `Property_Area`: Kırsal (`Rural = 0`), Yarı Kentsel (`Semiurban = 1`), Şehir Merkezi (`Urban = 2`).
   - `Loan_Status`: Kredi Onaylandı (`Y = 1`), Kredi Reddedildi (`N = 0`).
3. **Eğitim ve Test Ayrımı:**
   - Veri seti, sınıf oranlarını korumak için **Stratified** olarak **%80 Eğitim (491 satır)** ve **%20 Test (123 satır)** şeklinde ayrıldı.
4. **Özellik Ölçekleme (StandardScaler):**
   - Gelir (5000 birim) ile Kredi Geçmişi (0-1 birim) arasındaki sayısal ölçek farkını gidermek ve model katsayılarının birbiriyle kıyaslanabilmesini sağlamak amacıyla standartlaştırma uygulandı.

---

## 4. MODEL PERFORMANS VE DEĞERLENDİRME SONUÇLARI

Lojistik Regresyon modelinin test seti (123 başvuru) üzerindeki sonuçları:

| Metrik | Değer | Anlamı ve Yorumu |
| :--- | :---: | :--- |
| **Doğruluk (Accuracy)** | **%85.37** | Model tüm test başvurularının yaklaşık %85'ini doğru tahmin etmiştir. |
| **F1-Skoru (Onay)** | **%90.32** | Modelin kredi onayı sınıfındaki başarı dengesi mükemmel seviyededir. |
| **Kesinlik (Precision)** | **%83.17** | Modelin "Onaylanır" dediği müşterilerin %83'ü gerçekten onaylanmıştır. |
| **Duyarlılık (Recall)** | **%98.82** | Krediyi hak eden 85 müşterinin **84 tanesi** yakalanmış, müşteri kaybı önlenmiştir. |
| **ROC-AUC Skoru** | **0.8248** | 0.80'in üzerindeki AUC değeri modelin riskli ile güveniliri ayırma gücünün çok yüksek olduğunu gösterir. |

### Karmaşıklık Matrisi (Confusion Matrix):
```
                 Tahmin: Ret (0)  Tahmin: Onay (1)
Gerçek: Ret (0)         21               17          (Toplam: 38 Ret)
Gerçek: Onay (1)        1                84          (Toplam: 85 Onay)
```
- **Doğru Onaylananlar (TP):** 84 kişi.
- **Doğru Reddedilenler (TN):** 21 kişi.
- **Haksız Reddedilenler (FN):** Sadece 1 kişi (Model güvenilir müşterileri kaçırmamaktadır).
- **Hatalı Onay Verilenler (FP):** 17 kişi.

---

## 5. PARAMETRE ETKİ ANALİZİ (SONUCA ETKİ EDEN FAKTÖRLER)

Lojistik Regresyon katsayıları ($\beta$) ve Odds Ratio ($e^\beta$) değerleri, 6 parametrenin kredi kararına etkisini net bir şekilde ortaya koymaktadır:

| Sıra | Parametre | Model Katsayısı ($\beta$) | Odds Ratio ($e^\beta$) | Etki Yönü | Finansal ve İstatistiksel Açıklama |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **`Credit_History`** | **+1.2036** | **3.332x** | **Çok Güçlü Pozitif** | **En belirleyici faktördür.** Temiz bir kredi siciline sahip olmak, kredi onay şansını doğrudan **3.33 kat** artırmaktadır. Geçmişi sorunlu olanların büyük çoğunluğu elenmektedir. |
| **2** | **`Education`** | **+0.1460** | **1.157x** | Pozitif | Üniversite mezunu olmak onay alma olasılığını **%15.7** artırmaktadır (finansal okuryazarlık ve düzenli iş güvencesi göstergesi). |
| **3** | **`ApplicantIncome`** | **+0.0542** | **1.056x** | Pozitif | Gelir arttıkça ödeme gücü arttığı için onay ihtimali pozitif yönde yükselmektedir. |
| **4** | **`Property_Area`** | **+0.0431** | **1.044x** | Pozitif | Yarı kentsel ve kentsel bölgelerdeki gayrimenkullerin likiditesi kırsala göre daha yüksek olduğundan onay şansını hafif artırır. |
| **5** | **`Loan_Amount_Term`** | **-0.0009** | **0.999x** | **Nötr / Etkisiz** | Matematiksel olarak -0.0009 gibi sıfıra çok yakın bir değer alsa da, Odds Ratio tam 0.999x (~1.000) olduğu için istatistiksel ve finansal olarak **Nötr (Etkisiz)** kabul edilir. Başvuruların büyük çoğunluğu (%85+) standart 360 ay olduğundan vade süresi onay kararını tek başına etkilememektedir. |
| **6** | **`LoanAmount`** | **-0.0762** | **0.927x** | **Negatif** | **Ters etkilidir.** Talep edilen kredi tutarı arttıkça bankanın üstlendiği anapara riski arttığı için onay şansı her birimde **%7.3 azalmaktadır**. |

---

## 6. SONUÇ VE AKADEMİK DEĞERLENDİRME

1. **Bankacılık Mantığıyla Birebir Örtüşme:** Lojistik Regresyon katsayıları finansal mantıkla %100 tutarlıdır. Bankanın en çok önem verdiği unsur geçmiş borç sadakati (`Credit_History`) iken, en çok çekindiği unsur talep edilen yüksek kredi tutarıdır (`LoanAmount`).
2. **Yüksek Başarım ve Şeffaflık:** %85.37 doğruluk oranı ile model hem güvenilir tahminler yapmakta hem de her bir kararın arkasındaki gerekçeyi matematiksel olarak sunabilmektedir.
3. **Özet:** 6 parametre ve Lojistik Regresyon algoritması, kredi onay sınıflandırma problemi için hem teorik hem de pratik açıdan en uygun, en sade ve en başarılı çözümü sağlamıştır.

---

## 7. KAYNAKÇA VE REFERANSLAR (REFERENCES)

1. **altruistdelhite04.** (2021). *Loan Prediction Problem Dataset*. Kaggle.  
   Erişim adresi: [https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
2. **Thomas, L. C., Crook, J. N., & Edelman, D. B.** (2002). *Credit Scoring and its Applications*. SIAM - Society for Industrial and Applied Mathematics.
3. **Hand, D. J., & Henley, W. E.** (1997). Statistical classification methods in consumer credit scoring: a review. *Journal of the Royal Statistical Society: Series A (Statistics in Society)*, 160(3), 523-541.
4. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2nd Edition, Springer Series in Statistics.
