#Montando a pasta no Google Drive
import os
import torch
import shutil
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np

# Determinar o dispositivo (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Montando a pasta dos dados GoogleDrive - Colab
def montar_drive():
    try:
        drive.mount('/content/drive')
        print("Google Drive montado com sucesso!")
    except Exception as e:
        print("Não está no Colab ou Drive já montado:", e)

    data_dir = "/content/drive/MyDrive/DeepLearning/data_colab/fusca"

    # Cria a pasta caso não exista
    os.makedirs(data_dir, exist_ok=True)
    print("Pasta de dados:", data_dir)
    return data_dir

# Montando a pasta dos dados Local
def montar_pasta_local():
    data_dir = "data_colab/fusca"
    print("Pasta de dados local:", data_dir)
    return data_dir

# Função para separação em conjunto de teste e treino
def split_train_val(root_dir, train_ratio=0.8, seed=42):
    random.seed(seed)

    train_dir = root_dir + "_split/train"
    val_dir = root_dir + "_split/val"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [os.path.join(class_path, f) for f in os.listdir(class_path)]
        images = [f for f in images if os.path.isfile(f)]
        random.shuffle(images)

        n_train = int(len(images) * train_ratio)

        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        for img in images[:n_train]:
            shutil.copy(img, os.path.join(train_class_dir, os.path.basename(img)))
        for img in images[n_train:]:
            shutil.copy(img, os.path.join(val_class_dir, os.path.basename(img)))

        print(f"{class_name}: {n_train} train, {len(images)-n_train} val")

    return train_dir, val_dir

# Classe para ler as amostras
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []

        for idx, cls in enumerate(self.classes):
            class_dir = os.path.join(root_dir, cls)
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    self.image_paths.append(os.path.join(class_dir, img_file))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define o Modelo
def create_model(num_classes=6, pretrained=True):
    # Carrega ResNet18 com pesos do ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

    # Congela camadas iniciais
    for param in model.parameters():
        param.requires_grad = False

    # Substitui a camada final (fc) para o número de classes do dataset
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model

# Função de Treino e Validação
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\n Época {epoch+1}/{num_epochs}")
        print("-" * 30)

        # ---------- Treinamento ----------
        model.train()
        train_loss, correct, total = 0, 0, 0

        for images, labels in tqdm(train_loader, desc="Treinando", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = train_loss / total

        # ---------- Validação ----------
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validando", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss = val_loss / val_total

        # ---------- Logs ----------
        print(f"Treino → Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Validação → Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # ---------- Salvar melhor modelo ----------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Modelo salvo (melhor até agora).")

    print(f"\n Treinamento concluído. Melhor acurácia de validação: {best_val_acc:.4f}")
    return model

# Função de Avaliação
def evaluate_model(model, val_loader, class_names):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # ---- Acurácia geral ----
    acc = accuracy_score(y_true, y_pred)
    print(f"\n Acurácia total: {acc:.4f}\n")

    # ---- Relatório detalhado ----
    print("Relatório de classificação:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ---- Matriz de confusão ----
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.title("Matriz de Confusão")
    plt.show()

# Função de Predição (Inferência)
def predict_image(model, image_path, class_names, transform):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        pred_class = class_names[pred_idx]
        confidence = probs[0][pred_idx].item()

    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Predição: {pred_class} ({confidence*100:.1f}%)")
    plt.show()

    return pred_class, confidence

# Função principal
if __name__ == "__main__":
    root_dir = montar_pasta_local()
    class_names = ['40-59', '60-66', '67-72', '73-78', 'S. Ouro', 'Ultima s.']
    num_classes = 6
    num_epochs = 10

    # Transformações - Data Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Divide os dados em treino/validação
    train_dir, val_dir = split_train_val(root_dir)

    # Cria datasets
    train_dataset = CustomDataset(train_dir, transform=train_transform)
    val_dataset = CustomDataset(val_dir, transform=val_transform)

    # Cria DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    # Testa se está tudo certo
    for images, labels in train_loader:
        print(f"Lote de imagens: {images.shape}")
        print(f"Lote de labels: {labels}")
        break

    # Setup do Modelo e Otimizador
    model = create_model(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss() # Função de Custo e Otimização
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)  # só treina a cabeça da rede

    # Carregar modelo se existir
    model_path = "best_model.pth"

    # Treinamento
    if os.path.exists(model_path):
        print(f"\nModelo encontrado em '{model_path}'. Carregando pesos...")
        # Carrega os pesos salvos no modelo
        model.load_state_dict(torch.load(model_path, map_location=device))
        model_trained = model
    else:
        print("\nNenhum modelo encontrado. Iniciando Treinamento...")
        model_trained = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

    # Avaliação
    print('\nAvaliação Final:')
    evaluate_model(model, val_loader, class_names)

    # Teste e Inferência
    print("\nTeste de Inferência...")
    test_image_path = "data_colab/data_inferencia/treino_b.jpg"
    predict_image(model_trained, test_image_path, class_names, val_transform)