# MNIST-Real-Time-Digit-Recognition
## Projenin Amacı
Bu proje, **MNIST** el yazısı rakam verisini kullanarak bir CNN modeli eğitmek ve ardından gerçek zamanlı kamera görüntüsü üzerinden rakam tahmini yapmak amacıyla geliştirilmiştir.

İki ayrı script bulunmaktadır: 
- **MODELTRAIN.py** → CNN modelini eğitir ve kaydeder.
- **MODELPREDICT.py** → Kaydedilmiş modeli yükler ve kamera görüntüsü üzerinden rakam tahmini yapar.

## 💻 Kullanılan Teknolojiler
- Python 3.11.8
- PyTorch: Model oluşturma, eğitim ve tahmin için.
- Torchvision: MNIST dataset ve transform işlemleri için.
- OpenCV: Kamera görüntüsü yakalamak için
- Matplotlib: Eğitim verilerini ve tahmin sonuçlarını görselleştirmek için.
- TQDM: Eğitim sürecinde ilerleme çubuğu göstermek için.
- Timeit: Eğitim süresini ölçmek için.

## ⚙️ Kurulum
GEREKLİ KÜTÜPHANELERİ KURUN
```bash
pip install torch torchvision matplotlib opencv-python tqdm
```

## 🚀 Çalıştırma
1. Önce modeli eğitmek için **ModelTrain.py** dosyasını çalıştırın:
```bash
python ModelTrain.py
```
2. Eğitilmiş modeli kullanarak kamera üzerinden tahmin yapmak için **ModelPredict.py** dosyasını çalıştırın:
```bash
python ModelPredict.py
```
## BU PROJE HİÇBİR ŞEKİLDE TİCARİ AMAÇ İÇERMEMEKTEDİR.
