# VGLGUI

## Configuração do ambiente para o interpretador

## Instalação Drivers CPU

> Fazer instalação do AMDAPPSDK



## Instalação Drivers GPU

### Baixar drivers da amdgpu - site amd

> `sudo apt update`
> `sudo apt upgrade`
> `sudo apt clean`


### Instalar dependências
> `sudo apt install software-properties-common build-essential lsb-core dkms htop nmon psensor cpufrequtils cputool ipmitool ipmiutil smartmontools vainfo intel-gpu-tools mesa-opencl-icd mesa-utils-extra libegl1-mesa libgl1-mesa-glx libgles2-mesa libassimp5 ttf-mscorefonts-installer v4l-utils vim git p7zip-full p7zip-rar mesa-utils`


> `sudo apt update`
> `sudo apt install mesa-vulkan-drivers vulkan-tools libassimp5 libvulkan1`
> `sudo amdgpu-install --usecase=graphics,opencl --opencl=rocr,legacy --vulkan=amdvlk,pro`
> `sudo reboot`

### Verificar se instalou corretamente

> `sudo vulkaninfo | less`
> `sudo glxinfo | less`
> `sudo glxgears`
> `sudo clinfo`

### Outra alternativa
> `amdgpu-install --usecase=opencl --no-dkms`







## Instalação da Venv (Ambiente Virutal)
#### Baixar versão correspondente do python
 > `sudo apt install python3.10-venv`
 
### Criar ambiente virtual

 > `python3.10 -m venv my_env`
 
### Ativar ambiente virutal

 > `source my_env/bin/activate`
 
### Instalar dependências

>  `pip install pyopencl scikit-image matplotlib tifffile`



### Para executar a VGLGUI

> `make -f Makefile_python rundemo`

> `make -f Makefile_python runfundus`




#### Configurações para rodar GPU e CPU

Para rodar na CPU:

abrir o bash para alterar o LD_LIBARY_PATH

> nano ~/.bashrc //Comando para abrir
> export LD_LIBRARY_PATH="/opt/AMDAPPSDK-2.9-1/lib/x86_64/:/opt/amdgpu/lib/x86_64>   //Adicionar no final
> source ~/.bashrc //para atualizar
> exec bash //no terminal que for compilar o interpretador



Para rodar na GPU:
> nano ~/.bashrc //Comando para abrir
> export LD_LIBRARY_PATH="/opt/amdgpu/lib/x86_64-linux-gnu/:/opt/rocm/lib/"
> source ~/.bashrc //para atualizar
> exec bash //no terminal que for compilar o interpretador
