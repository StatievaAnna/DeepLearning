import os
from PIL import Image
import numpy as np
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
from torch.nn.functional import pairwise_distance
from sklearn.metrics import f1_score, roc_curve, auc
from tqdm import tqdm
from matplotlib import pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FaceDetection():
    def __init__(self):
        # Модель FaceNet 
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.model.train()

        # Аугментация 
        self.augment = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            # transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5])
        ])
        
        # MTCNN для обрезки лиц  
        self.mtcnn = MTCNN(image_size=160, margin=10, keep_all=False, device=device)


    # Извлечение эмбеддинга  
    @torch.no_grad()
    def get_embedding(self, image_path):
        self.model.eval()
        image = Image.open(image_path).convert("RGB")
        image = self.augment(image)
        face = self.mtcnn(image)
        if face is None:
            # raise ValueError(f"Лицо не найдено: {image_path}")
            face = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])(image).unsqueeze(0).to(device)
        else:
            face = transforms.Compose([transforms.Resize((160, 160))])(face).unsqueeze(0).to(device)
        emb = self.model(face)
        return emb.squeeze(0)


    #  Получить эмбеддинги для всех изображений в папке  
    def load_embeddings(self, folder, label):
        embeddings, labels = [], []
        for filename in os.listdir(folder):
            if filename.startswith('.'): continue
            path = os.path.join(folder, filename)
            try:
                emb = self.get_embedding(path)
                embeddings.append(emb)
                labels.append(label)
            except Exception as e:
                print(f"Ошибка с {path}: {e}")
        return embeddings, labels
    
    
    #   Загружаем пары "same" и "different"  
    def load_pairs(self, anchor_image_path, same_folder, diff_folder):
        print("Предобработка изображений...")
        
        distances = []
        labels = []
        
        files = os.listdir(same_folder)
        same_file = os.path.join(same_folder, files[0])
        emb_anchor = self.get_embedding(anchor_image_path)

        # Пары "same"
        for f in files:
            if f.startswith('.'): continue  # игнор .DS_Store
            emb1 = self.get_embedding(os.path.join(same_folder, f))
            dist = pairwise_distance(emb_anchor.unsqueeze(0), emb1.unsqueeze(0)).item()
            distances.append(dist)
            labels.append(1)  # same


        # Пары "different"
        for f in os.listdir(diff_folder):
            if f.startswith('.'): continue  # игнор .DS_Store
            emb2 = self.get_embedding(os.path.join(diff_folder, f))
            dist = pairwise_distance(emb_anchor.unsqueeze(0), emb2.unsqueeze(0)).item()
            distances.append(dist)
            labels.append(0)  # different

        return np.array(distances), np.array(labels)


    # Подбор оптимального threshold  
    def find_best_threshold(self, distances, labels):
        print("Подбор оптимального параметра...")
        
        thresholds = np.linspace(distances.min(), distances.max(), 100)
        best_f1 = 0
        best_thresh = 0

        for t in tqdm(thresholds):
            preds = (distances < t).astype(int)
            score = f1_score(labels, preds)
            if score > best_f1:
                best_f1 = score
                best_thresh = t
                
        best_preds = (distances < best_thresh).astype(int)

        print(f"Лучший threshold: {best_thresh:.4f}, F1: {best_f1:.4f}")
        
        tn = fp = fn = tp = 0
        for pred, target in zip(best_preds, labels):
            if pred == 1:
                if target == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if target == 1:
                    fn += 1
                else:
                    tn += 1

        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn
        
        self.f1 = best_f1
        
        return best_thresh


    # Построение ROC-кривой  
    def plot_roc(self, distances, labels, name):
        print("построение ROC-кривой...")
        
        fpr, tpr, thresholds = roc_curve(labels, -distances)  # минус для "ближе — лучше"
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        label = f'ROC curve (AUC = {roc_auc:.2f})\nF1-score = {round(self.f1,2)}\nTP = {self.tp}\nTN = {self.tn}\nFP = {self.fp}\nFN = {self.fn}'
        plt.plot(fpr, tpr, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC-кривая Face Verification ({name})')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"ROC_{name}")
        plt.show()