FROM python:3.11

# Etiqueta de autor
LABEL authors="santhiperbolico"

# Configurar el directorio de trabajo
WORKDIR /hvs

# Copiar el código fuente
COPY src/ /hvs/

# Establecer la variable de entorno para PYTHONPATH
ENV PYTHONPATH="/hvs"

# Instalar dependencias
RUN pip install -r requirements-gcp.txt
