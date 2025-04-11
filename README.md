# Computer Vision YOLOv11 Project

Esta es una implementación de YOLOv11 para tareas de visión por computador.

## Configuración del Entorno

Sigue estos pasos para configurar tu entorno de desarrollo y poder ejecutar el proyecto.

### 1. Clonar el Repositorio

Primero, clona este repositorio en tu máquina local usando Git:

```bash
git clone https://github.com/johncortes117/computer-vision-YOLOv11.git
```

### 2. Navegar al Directorio del Proyecto

Una vez clonado, entra al directorio del proyecto:

```bash
cd computer_vision_YOLOv11
```

### 3. Crear un Entorno Virtual

Es recomendable usar un entorno virtual para aislar las dependencias del proyecto. Crea uno usando `venv`:

```bash
python -m venv venv
```

Esto creará una carpeta `venv` dentro del directorio del proyecto.

### 4. Activar el Entorno Virtual

Activa el entorno virtual recién creado. El comando varía según tu sistema operativo:

*   **Windows:**
    ```bash
    .\venv\Scripts\activate
    ```
*   **macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```

Una vez activado, verás `(venv)` al principio de la línea de comandos.

### 5. Instalar Dependencias

Instala todas las librerías necesarias que se encuentran listadas en el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Ejecutar el Programa

Ejecuta el programa utilizando el siguiente comando.

```bash
python main.py
```