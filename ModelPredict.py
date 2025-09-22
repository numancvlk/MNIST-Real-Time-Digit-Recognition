import cv2 as cv

import torch
from torch import nn

from torchvision import transforms

from collections import deque #Kararlı tahminler için FIFO (İlk Giren İlk Çıkar) kuyruk yapısı.

class MNISTMODEL(nn.Module):
    def __init__(self, inputShape=1, hiddenUnit=10, outputShape=10):
        super().__init__()
        self.convStack1 = nn.Sequential(
            nn.Conv2d(inputShape, hiddenUnit, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hiddenUnit, hiddenUnit, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.convStack2 = nn.Sequential(
            nn.Conv2d(hiddenUnit, hiddenUnit, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hiddenUnit, hiddenUnit, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hiddenUnit*7*7, outputShape)
        )
    def forward(self, x):
        x = self.convStack1(x)
        x = self.convStack2(x)
        x = self.classifier(x)
        return x

#----------KAYDEDİLEN MODELİ YÜKLEMEK----------
device = "cuda" if torch.cuda.is_available() else "cpu"
trainedModel = MNISTMODEL(1, 64, 10).to(device)
trainedModel.load_state_dict(torch.load("Source/myMNISTMODEL.pth", map_location=device))
#----------KAYDEDİLEN MODELİ YÜKLEMEK----------


trainedModel.eval()

# Görüntüleri modele girmeden önce dönüştürmek için transform tanımlıyoruz
transform = transforms.Compose([
    transforms.ToTensor(), # Görüntüyü PyTorch tensörüne çevirir (0-1 aralığında)
    transforms.Normalize((0.5,), (0.5,)) ## Normalizasyon: ortalama 0.5, std 0.5 GRİ OLDUĞU İÇİN RESİMLER BÖYLE RGB OLSA 3 TANE ORT 3 TANE STD DEĞERİ OLURDU
])

#Son 5 tahmini saklayacak bir kuyruk oluşturuyoruz. 
#DAHA İYİ TAHMİNLER YAPABİLMEK İÇİN 
pred_queue = deque(maxlen=5)

## Tahmin güven eşiği; modelin sadece %80 üzerinde emin olduğu tahminleri kabul edeceğiz
confidence_threshold = 0.8  

capture = cv.VideoCapture(0) 

while True:
    ret, frame = capture.read() ## Kameradan bir kare oku
    if not ret: ## Eğer kare okunamazsa döngüyü kır
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) ## Kameradan gelen kareyi gri tonlamaya çeviriyoruz

    ## Kareyi önce blur ile yumuşatıyoruz, sonra binary (siyah-beyaz) hale getiriyoruz
    # ve morfolojik açma ile küçük gürültüleri temizliyoruz
    thresh = cv.morphologyEx(
        cv.threshold(cv.GaussianBlur(gray, (5,5), 0), 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1],
        cv.MORPH_OPEN, # Morfolojik açma: küçük beyaz gürültüleri temizler
        cv.getStructuringElement(cv.MORPH_RECT, (3,3)) ## 3x3 dikdörtgen kernel
    )

    # Görüntüdeki konturları buluyoruz (nesne sınırları)
    contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    if contours: ## Eğer kontur varsa (yani rakam veya nesne var)
        cnt = max(contours, key=cv.contourArea) ## En büyük konturu seçiyoruz, büyük ihtimalle rakam
        if 2000 < cv.contourArea(cnt) < 80000: ## Alan çok küçük veya çok büyükse göz ardı et
            x, y, w, h = cv.boundingRect(cnt) ## Konturu dikdörtgen içine alacak şekilde koordinatları alıyoruz
            
            # # ROI (Region of Interest) oluşturuyoruz, yani sadece rakam kısmı
            roi = cv.resize(gray[y:y+h, x:x+w], (28,28)) # Model input boyutu 28x28
            roi_tensor = transform(roi).unsqueeze(0).to(device) # Tensöre çevir ve batch dimension ekle

            with torch.no_grad():
                probs = torch.softmax(trainedModel(roi_tensor), dim=1) # Olasılıkları al
                pred_conf, pred_class = torch.max(probs, dim=1) # En yüksek olasılık ve sınıf
                pred_conf, pred_class = pred_conf.item(), pred_class.item() # Tensor → float/int
            
            if pred_conf >= confidence_threshold: # Eğer tahmin güveni belirlediğimiz eşik üstündeyse
                pred_queue.append(pred_class) ## Tahmini kuyruğa ekle
                stable_pred = max(set(pred_queue), key=pred_queue.count) # Son 5 tahminin en çok tekrar edeni al, böylece daha stabil tahmin elde et              # Son 5 tahminin en çok tekrar edeni al, böylece daha stabil tahmin elde et
                
                ## Kareye dikdörtgen çiz ve tahmini yaz
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                cv.putText(frame, f"{stable_pred} ({pred_conf*100:.0f}%)", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv.imshow("Camera", frame)
    if cv.waitKey(1) == 27:
        break

capture.release()
cv.destroyAllWindows()
