# MNIST-Real-Time-Digit-Recognition
# [TR]
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

## 📺 Uygulama Videosu
▶️ [Watch Project Video on YouTube](https://www.youtube.com/watch?v=RHs6ePOwDL8)
## BU PROJE HİÇBİR ŞEKİLDE TİCARİ AMAÇ İÇERMEMEKTEDİR.

# [EN]
## Project Purpose
This project aims to train a CNN model using the **MNIST** handwritten digit dataset and then make digit predictions through real-time camera input.

There are two separate scripts: 
- **MODELTRAIN.py** → Trains and saves the CNN model.
- **MODELPREDICT.py** → Loads the saved model and makes digit predictions through the camera.

## 💻 Technologies Used
- Python 3.11.8
- PyTorch: For building, training, and predicting with the model.
- Torchvision: For MNIST dataset and transform operations.
- OpenCV: For capturing camera input.
- Matplotlib: For visualizing training data and prediction results.
- TQDM: For displaying a progress bar during training.
- Timeit: For measuring training duration.

## ⚙️ Installation
INSTALL THE REQUIRED LIBRARIES
```bash
pip install torch torchvision matplotlib opencv-python tqdm
```

## 🚀 Run
1. First, run the **ModelTrain.py** file to train the model:
```bash
python ModelTrain.py
```
2. To make predictions through the camera using the trained model, run the **ModelPredict.py** file:
```bash
python ModelPredict.py
```
## 📺 Application Video
▶️ [Watch Project Video on YouTube](https://www.youtube.com/watch?v=RHs6ePOwDL8)
## THIS PROJECT DOES NOT INVOLVE ANY COMMERCIAL PURPOSE IN ANY WAY.
