# Yolo_pose - ROS Foxy

Este repositório contém um pacote de percepção utilizado na task "Follow Me" da RoboCup @Home, usando ROS2 Foxy.

## Descrição
Este pacote é uma implementação do YOLOv8 com TensorRT em ROS2. Ele é projetado para identificar pessoas em tempo real, publicar as poses desenhadas nas imagens capturadas e publicar o ponto central das poses do tronco. Essa funcionalidade é essencial para a task "Follow Me", permitindo que o robô siga a pessoa corretamente.



## Pré-requisitos

- Docker
- Imagem docker com ROS 2 Foxy
- Yolo_v8 engine para GPU específica

### Para buildar a imagem

 Siga esse repositório:
 https://github.com/triple-Mu/YOLOv8-TensorRT.git

 > Lembre que utilizamos o modelo para pose.

## Configuração

### Passo 1: Clonar o repositório e configurar-lo

```bash
git clone git@github.com:Pequi-Mecanico-Home/yolo_pose.git -b <branch>
cd yolo_pose
```

Em seguida adicione a engine dentro da pasta config.
> Se adicionar com o seguite nome não precisará de alterações: `yolov8n-pose.engine`.

### Passo 2: Obter a imagem Docker

Buildar um Dockerfile na raiz ou use o existente.

#### X86:

```bash
docker push alexandreacff/yollov8-ros-foxy:tensorrt-ros
```

### Passo 3: Rodar o container - linux

```bash
xhost local: && docker run -it --rm  \
                --user root \
                --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"  \
                --name yolo_pose-container --ipc=host --net=host  \
                --gpus all \
                --volume="$PWD:/root/dev_ws/src/yolo_pose" \
                --privileged \
                --ulimit memlock=-1 --ulimit stack=67108864 \
                -w /root/dev_ws \
                alexandreacff/yollov8-ros-foxy:tensorrt-ros bash
```
### Passo 4: Abrir outros terminais do container - preferência 5.

```bash
docker exec -it yolo_pose-container bash
```

### Passo 5: Iniciar o pacote de percepção

No container, inicie o pacote de percepção para rastreamento de pessoas.

#### Exemplo de comando para iniciar o pacote:

```bash
source /opt/ros/foxy/setup.bash
colcon build --symlink-install
source /ros2_ws/install/setup.bash
ros2 launch realsense2_camera rs_camera.launch.py
```

Em outros terminal:

```bash
source /opt/ros/foxy/setup.bash
source /ros2_ws/install/setup.bash
ros2 run yolo_pose inference
```

```bash
source /opt/ros/foxy/setup.bash
source /ros2_ws/install/setup.bash
ros2 run yolo_pose draw_keypoints
```

```bash
source /opt/ros/foxy/setup.bash
source /ros2_ws/install/setup.bash
ros2 run yolo_pose distance
```
