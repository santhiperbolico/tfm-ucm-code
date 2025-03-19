FROM python:3.11

# Etiqueta de autor
LABEL authors="santhiperbolico"

# Configurar el directorio de trabajo
WORKDIR /hvs

COPY src/requirements-gcp.txt /hvs/requirements-gcp.txt
COPY src/requirements.txt /hvs/requirements.txt

# Instalar dependencias
RUN pip install -r requirements-gcp.txt

# Copiar el código fuente
COPY src/hyper_velocity_stars_detection/ /hvs/hyper_velocity_stars_detection/

# Establecer la variable de entorno para PYTHONPATH
ENV PYTHONPATH="/hvs"
