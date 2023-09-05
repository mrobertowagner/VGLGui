# VGLGUI - Interpreter Workflow

## Configuração do ambiente para o interpretador

## Instalação Drivers CPU (AMD e INTEL)

> Fazer instalação do driver AMDAPPSDK

* [AMDAPPSDK](https://github.com/ghostlander/AMD-APP-SDK/releases/tag/v2.9.1)

> Abrir na pasta do AMDAPPSDK e digitar no terminal ./install.sh

### Para confirmar se instalou corretamente:
```
clinfo
```

## Instalação Drivers GPU (AMD)

### Baixar drivers da amdgpu 
```
 sudo apt update && sudo apt upgrade && sudo apt clean
```

### Instalar dependências

```
sudo apt install software-properties-common build-essential lsb-core dkms htop nmon psensor cpufrequtils cputool ipmitool ipmiutil smartmontools vainfo intel-gpu-tools mesa-opencl-icd mesa-utils-extra libegl1-mesa libgl1-mesa-glx libgles2-mesa libassimp5 ttf-mscorefonts-installer v4l-utils vim git p7zip-full p7zip-rar mesa-utils
```

```
sudo apt update
```
```
sudo apt install mesa-vulkan-drivers vulkan-tools libassimp5 libvulkan1
```
```
sudo amdgpu-install --usecase=graphics,opencl --opencl=rocr,legacy --vulkan=amdvlk,pro
```
```
sudo reboot
```
### Verificar se instalou corretamente

```
sudo clinfo
```
### Outra alternativa para instalação dos drivers GPU AMD
```
amdgpu-install --usecase=opencl --no-dkms
```


## Instalação da Venv (Ambiente Virutal)
#### Baixar versão correspondente do Python
```
sudo apt install python3.10-venv
```

### Criar ambiente virtual
```
python3.10 -m venv my_env
 ```
### Ativar ambiente virutal
```
source my_env/bin/activate
```
### Instalar dependências
```
pip install pyopencl scikit-image matplotlib tifffile
```


## Executar Interpretador

#### Executar o WorkFlow Demo
```
make -f Makefile_python rundemo
```
#### Executar o WorkFlow Fundus
```
make -f Makefile_python runfundus
```


## Configurações para rodar GPU e CPU em alguns casos:

### Para rodar na CPU:

### Abrir o bash para alterar o LD_LIBARY_PATH
```
nano ~/.bashrc 
```
### Adicionar no final do arquivo:
```
export LD_LIBRARY_PATH="/opt/AMDAPPSDK-2.9-1/lib/x86_64/:/opt/amdgpu/lib/x86_64>
```

### Para rodar na GPU:
```
export LD_LIBRARY_PATH="/opt/amdgpu/lib/x86_64-linux-gnu/:/opt/rocm/lib/"
```



## Referências

* [A System Architecture in Multiple Views for an Image Processing Graphical User Interface](https://www.researchgate.net/publication/351340432_A_System_Architecture_in_Multiple_Views_for_an_Image_Processing_Graphical_User_Interface?_sg%5B0%5D=lNjR7sqxLkU2nuIWM6Q91G3lXa9l_Op-YZUuFBaxyrt3aBeveZWALmZFbh064E14UrghKrbiYc91kZMKdm6wCoNnfTtP_mhypRjCO4RC.bikdAVOJj-vDeLJWcMvQdcYF1bhps9Oy5UKmT1Ng4uvCAp7zyTKk70TJx1EeFe4HPXFm-JUBzmJ5h_lQtRXHAw&_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6Il9kaXJlY3QiLCJwYWdlIjoicHJvZmlsZSIsInByZXZpb3VzUGFnZSI6InNlYXJjaCJ9fQ) - Artigo da arquitetura VGLGui.

* [High-Level Workflow Interpreter for Real-Time Image Processing](https://www.researchgate.net/publication/369016131_High-Level_Workflow_Interpreter_for_Real-Time_Image_Processing) - Artigo do VGLGui módulo do interpretador.
