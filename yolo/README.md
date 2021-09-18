## 05 - Detecção de Objetos com YOLO e OpenCV

Neste projeto é feito a detecção de um objetos em uma imagem. O objetos são detectados pelo YOLO e em seguida tem seus objetos destacados utilizando OpenCV.

### O que é detecção de objetos
A diferença entre métodos de classificação de objetos e métodos de detecção de objetos, é que na detecção de objetos você deseja:

* identificar quantos objetos de uma determinada categoria se encontram na imagem;
* identificar onde na imagem cada objeto se encontra.


### Requisitos
Os recursos necessarios para realizar a construção desse projeto foram:
    
    * Python3, foi a linguagem utilizada para desenvolver o projeto.
    * OpenCV, uma das bibliotecas mais conhecidas para desenvolvimento de Visão Computacional;    
    * YOLOv4
Baixe os pesos do YOLOv4 para a pasta weights, os comandos abaixo são no terminal:

``` shell
# Entre na pasta
cd weights

# Baixe os pesos, yolov4.weights > 200MB
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

```
### You only look once (YOLO)
Outros sistemas de detecção adaptam classificadores ou localizadores para realizar a detecção. Eles aplicam o modelo a uma imagem em vários locais e escalas. As regiões de alta pontuação da imagem são consideradas detecções.

O YOLO utiliza uma abordagem totalmente diferente. Aplicando uma única rede neural à imagem completa. Essa rede divide a imagem em regiões e prevê caixas delimitadoras e probabilidades para cada região. Essas caixas delimitadoras são ponderadas pelas probabilidades previstas.

Exemplos de uma detecção utilizando YOLO:

<img src="/Figuras/deteccao_image_1.jpg"/> 
<img src="/Figuras/deteccao_image_2.jpg"/>

### Para executar
Digite no terminal o comando abaixo:
```
python YOLO.py --image image_path
```
Exemplo de execução:
```
python YOLO.py --image dogs.jpg
```
Argumentos:

| Argumentos | Value Default          | Significado                                  | Obrigatorio |
|------------|------------------------|----------------------------------------------|-------------|
| --image    | image.jpg              | Caminho da imagem para se detectar objetos   | Sim         |
| --weights  | weights/yolov4.weights | Caminho para os pesos do YOLO4               | Sim         |
| --cfg      | cfg/yolov4.cfg         | Caminho para as configurações da CNN do YOLO | Sim         |
| --labels   | cfg/coco.names         | Caminho para as configurações da CNN do YOLO | Sim         |
| --resize   | True                   | Realizar redimensionamento da imagem para imagens muito grandes | Opcional|
| --show_text | False                 | Mostra saídas                                | Opcional    |
| --show_image| False                 | Mostra a imagem ao final da detecção         | Opcional    |
| --threshold| 0.5                    | Threshold para probabilidade de existir um objeto| Opcional    |
| --threshold_NMS| 0.3                | Threshold para NON-MAX SUPRESSION | Opcional    |


## Referências

YOLO: <a href="url">https://pjreddie.com/darknet/yolo/</a>