# Projeto-ClassificadorIA
Projeto de um Classificador de imagens usando IA

## Descrição do Projeto

Este projeto implementa um modelo de classificação de imagens de Fuscas em diferentes categorias usando Transfer Learning com a arquitetura **ResNet18** do PyTorch.
O objetivo é treinar e avaliar um modelo capaz de identificar corretamente a faixa de ano ou categoria de um Fusca a partir de imagens.

O código foi estruturado para funcionar tanto no Google Colab (com Google Drive) quanto em ambiente local.

⚙️ Funcionalidades Principais:
 - Montagem automática do Google Drive (Colab)
 - Divisão de dados em treino e validação (80/20)
 - Data Augmentation com torchvision.transforms
 - Treinamento supervisionado com ResNet18 pré-treinada no ImageNet
 - Avaliação com matriz de confusão e relatório de classificação
 - Inferência visual com confiança (%)

🧩 Dependências:
 - torch
 - torchvision
 - pillow
 - tqdm
 - matplotlib
 - seaborn
 - scikit-learn
 - numpy
