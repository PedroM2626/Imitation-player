# Comparacao Completa de Modelos: 6 Arquiteturas

## Resultados

| Metrica             | NatureCNN          | CNN_LSTM           | ViT_Transformer    | Impoola_CNN        | Impala_CNN         | ResNet18           |
|---------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| Loss Final          | 3.476316           | 3.024518           | 2.998286           | 2.933126           | 3.027630           | 2.901681           |
| Tempo de Treino     | 18.3            s | 67.8            s | 3288.8          s | 226.2           s | 345.1           s | 251.3           s |
| Tamanho do Modelo   | 16.33           MB | 23.71           MB | 9.70            MB | 4.20            MB | 67.95           MB | 44.37           MB |
| Num. Parametros     | 4,196,810         | 6,116,779         | 2,448,010         | 1,009,258         | 17,720,938        | 11,516,938        |

## Analise

- **Menor Loss**: ResNet18
- **Mais Rapido**: NatureCNN
- **Mais Leve**: Impoola CNN

### Descricao das Novas Arquiteturas Adicionadas:
*   **Impala-CNN**: Modelo clássico do artigo do IMPALA. Apresenta ótima capacidade de memorização espacial por reter o Flatten da camada convolucional, porém gera um arquivo muito pesado.
*   **ResNet-18**: Arquitetura convolucional profunda com conexões residuais padrão. Oferece boa estabilidade de gradientes, mas é redundante para resoluções de 128x128.

### Nota sobre Swin Transformer e ConvNeXt:
Tanto o **Swin Transformer** quanto o **ConvNeXt (Tiny)** padrão foram avaliados, mas **removidos do script ativo de treinamento** devido à extrema lentidão de processamento e alto consumo de hardware. 
O ConvNeXt (Tiny) e o Swin Transformer possuem cerca de 28 milhões de parâmetros cada, o que provoca gargalo de memória de vídeo (VRAM Thrashing) ao processar lotes de 384 em imagens empilhadas, resultando em tempos estimados de treino de **40 horas** (no caso do ConvNeXt) no hardware de desenvolvimento, inviabilizando testes locais ágeis. Embora sejam arquiteturas extremamente potentes para visão geral, exigem recursos de cluster ou GPUs com alta memória dedicada.

## Configuracao do Experimento

- Epocas de Treino: 10
- Batch Size: 384
- Learning Rate: 0.0001
- Device: cuda

## Artefatos
- Modelos salvos na pasta `./models/`
- Experimento MLflow: `Model_Comparison`
