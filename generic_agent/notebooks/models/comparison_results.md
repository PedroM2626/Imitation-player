# Comparacao de Modelos: NatureCNN vs CNN+LSTM vs ViT Transformer

## Resultados

| Metrica             | NatureCNN          | CNN+LSTM           | ViT Transformer    |
|---------------------|--------------------|--------------------|--------------------|
| Loss Final          | 3.476316          | 3.024518          | 2.998286          |
| Tempo de Treino     | 18.3s               | 67.8s               | 3288.8s               |
| Tamanho do Modelo   | 16.33 MB          | 23.71 MB          | 9.70 MB          |
| Num. Parametros     | 4,196,810        | 6,116,779        | 2,448,010        |

## Analise

- **Menor Loss**: ViT Transformer
- **Mais Rapido**: NatureCNN
- **NatureCNN**: Modelo convolucional padrao, mais leve e rapido de treinar.
- **CNN+LSTM**: Adiciona memoria temporal com LSTM bidirecional e atencao temporal.
- **ViT Transformer**: Modelo baseado em Self-Attention, captura relacoes espaciais e temporais de forma mais sofisticada, mas exige mais recursos.

## Configuracao do Experimento

- Epocas: 10
- Batch Size: 384
- Learning Rate: 0.0001
- Device: cuda
- Num. Acoes: 9

## Artefatos

- Modelo NatureCNN: `models/NatureCNN_policy.zip`
- Modelo CNN+LSTM: `models/CNN_LSTM_policy.zip`
- Modelo ViT: `models/ViT_Transformer_policy.zip`
- MLflow Experiment: `Model_Comparison`
