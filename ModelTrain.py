#LIBRARIES
import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

from timeit import default_timer

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import random

train_transform = transforms.Compose([
    transforms.RandomRotation(15),              # -15° ile +15° arası döndür
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # %10’a kadar kaydır
    transforms.RandomResizedCrop(28, scale=(0.9, 1.0)), # küçült-büyüt
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Test için augment gerekmez, sadece normalize
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#COLLECTING DATAS
trainData = datasets.MNIST(
    root="data",
    train=True,
    transform=train_transform,
    target_transform=None,
    download=True
)

testData = datasets.MNIST(
    root="data",
    train=False,
    transform=test_transform,
    target_transform=None,
    download=True
)

#DATALARI GÖRSELLEŞTİRMEK
fig = plt.figure(figsize=(10,10))
rows, cols = 5,5

for i in range(1,rows*cols+1):
    randomIndex = torch.randint(0,len(trainData),size=[1]).item()
    image, label = trainData[randomIndex]
    ax = plt.subplot(rows,cols,i)
    ax.set_title(trainData.classes[label])
    ax.imshow(image.squeeze())
    ax.axis("off")
plt.show()

#VERİLERİ DATA LOADERA EKLEMEK

BATCH_SIZE = 32

trainDataLoader = DataLoader(
    dataset=trainData,
    batch_size= BATCH_SIZE,
    shuffle=True
)

testDataLoader = DataLoader(
    dataset=testData,
    batch_size=BATCH_SIZE,
    shuffle=False
)


#MODELİ OLUŞTURMAK
class MNISTMODEL(nn.Module):
    def __init__(self,
                 inputShape: int,
                 hiddenUnit: int,
                 outputShape: int):
        super().__init__()

        self.convStack1 = nn.Sequential(
            nn.Conv2d(in_channels=inputShape,
                      out_channels=hiddenUnit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels= hiddenUnit,
                      out_channels= hiddenUnit,
                      kernel_size= 3,
                      stride= 1,
                      padding=1),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.convStack2 = nn.Sequential(
            nn.Conv2d(in_channels=hiddenUnit,
                      out_channels=hiddenUnit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hiddenUnit,
                      out_channels=hiddenUnit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hiddenUnit * 7 * 7,
                      out_features=outputShape)
        )

    def forward(self,x):
        x = self.convStack1(x)
        x = self.convStack2(x)
        x = self.classifier(x)
        return x
    

#MODELİN NESNESİNİ OLUŞTURMAK
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
myModel = MNISTMODEL(inputShape=1,
                     hiddenUnit=64,
                     outputShape=len(trainData.classes)).to(device)
print(myModel)

#MODELİ EĞİTMEK
def accuracy(yTrue,yPred):
    correct = torch.eq(yTrue,yPred).sum().item()
    acc = correct / len(yTrue)
    return acc

def printTrainTime(start,end,device):
    trainTime = end - start
    print(f"Train time is {trainTime} on the {device}")

lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=myModel.parameters(),
                            lr=0.1)

#TRAIN STEP
def trainStep(model: torch.nn.Module,
              dataset: torch.utils.data.DataLoader,
              lossFn: torch.nn.Module,
              accFn,
              device: torch.device = device):
    
    trainLoss, trainAcc = 0,0
    model.train()
    for batch, (xTrain,yTrain) in enumerate(dataset):
        xTrain, yTrain = xTrain.to(device), yTrain.to(device)

        #1 - FORWARD
        trainPred = model(xTrain)

        #2 - LOSS / ACC
        loss = lossFn(trainPred,yTrain)
        trainLoss += loss

        acc = accFn(yTrue = yTrain, yPred = trainPred.argmax(dim=1))
        trainAcc += acc

        #3 - ZERO GRAD
        optimizer.zero_grad()

        #4 - BACKWARD
        loss.backward()

        #5 - STEP
        optimizer.step()

        if batch % 400 == 0:
            print(f"Looked at {batch * len(xTrain)}/{len(dataset.dataset)} samples")
    
    trainLoss /= len(dataset)
    trainAcc /= len(dataset)

    print(f"Train Loss = {trainLoss:.5f} | Train Accuracy = {trainAcc:.5f}%")


def testStep(model: torch.nn.Module,
             dataset: torch.utils.data.DataLoader,
             lossFn: torch.nn.Module,
             accFn,
             device: torch.device = device):
    
    testLoss, testAcc = 0,0
    model.eval()

    for xTest,yTest in dataset:

        xTest, yTest = xTest.to(device), yTest.to(device)
        
        testPred = model(xTest)

        loss = lossFn(testPred,yTest)
        testLoss += loss

        acc = accFn(yTrue = yTest, yPred = testPred.argmax(dim=1))
        testAcc += acc

    testLoss /= len(dataset)
    testAcc /= len(dataset)

    print(f"Test Loss = {testLoss:.5f} | Test Accuracy = {testAcc:.5f}%")

def modelSummary(model:torch.nn.Module,
                 dataset: torch.utils.data.DataLoader,
                 lossFn: torch.nn.Module,
                 accFn,
                 device: torch.device = device):
    
    summaryLoss, summaryAcc = 0,0
    model.eval()

    for xTest, yTest in dataset:
        xTest, yTest = xTest.to(device), yTest.to(device)

        summaryPred = model(xTest)

        loss = lossFn(summaryPred, yTest)
        summaryLoss += loss

        acc = accFn(yTrue = yTest, yPred = summaryPred.argmax(dim=1))
        summaryAcc += acc

    summaryLoss /= len(dataset)
    summaryAcc /= len(dataset)

    return {"MODEL NAME": model.__class__.__name__,
            "MODEL LOSS": summaryLoss,
            "MODEL ACCURACY": summaryAcc}


torch.manual_seed(30)

epochs = 15

startTrainTimer = default_timer()

for epoch in tqdm(range(epochs)):
    trainStep(model=myModel,
              dataset=trainDataLoader,
              lossFn=lossFn,
              accFn=accuracy,
              device=device)
    
    testStep(model=myModel,
             dataset=testDataLoader,
             lossFn=lossFn,
             accFn=accuracy,
             device=device)

endTrainTimer = default_timer()

printTrainTime(start=startTrainTimer,
               end=endTrainTimer,
               device=device)

modelSum = modelSummary(model=myModel,
             dataset=testDataLoader,
             lossFn=lossFn,
             accFn=accuracy,
             device=device)

print(modelSum)

#PREDICTION YAPMAK
def makePredictions(model:torch.nn.Module,
                    data:list,
                    device:torch.device=device):

  predProbs = []
  model.to(device)
  model.eval()

  with torch.inference_mode():
    for sample in data:
      sample = torch.unsqueeze(sample,dim=0).to(device)

      predLogits = model(sample)

      predProb = torch.softmax(predLogits.squeeze(), dim=0)

      predProbs.append(predProb.cpu())

  return torch.stack(predProbs)



random.seed(41)
testSamples = []
testLabels = []

for sample, label in random.sample(list(testData), k=9):
  testSamples.append(sample)
  testLabels.append(label)


prediction = makePredictions(model=myModel,
                             data=testSamples,
                             device=device)

predictionClasses = prediction.argmax(dim=1)


# PLOT PREDICTIONS WITH COLORS
plt.figure(figsize=(9,9))
nrows = 3
ncols = 3

for i, sample in enumerate(testSamples):
    plt.subplot(nrows, ncols, i+1)

    # Görüntüyü göster
    plt.imshow(sample.squeeze(), cmap="gray")
    plt.axis('off')  # Eksenleri gizle

    # Tahmin ve gerçek sınıf isimleri
    predLabel = testData.classes[predictionClasses[i]]  # modelin tahmini
    trueLabel = testData.classes[testLabels[i]]         # gerçek sınıf

    # Renk belirle
    color = "green" if predLabel == trueLabel else "red"

    # Başlık ekle
    plt.title(f"P: {predLabel}\nT: {trueLabel}", color=color, fontsize=10)

plt.tight_layout()
plt.show()

torch.save(myModel.state_dict(),"myMNISTMODEL.pth")
print("Model ağırlıkları kaydedildi!")