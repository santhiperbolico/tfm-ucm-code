FROM python:3.11

# Etiqueta de autor
LABEL authors="santhiperbolico"

# Configurar el directorio de trabajo
WORKDIR /hvs

# Copiar el código fuente
COPY src/ /hvs/
COPY executables/mwgc.dat.txt /data/mwgc.dat.txt
COPY executables/ /hvs/

# Establecer la variable de entorno para PYTHONPATH
ENV PYTHONPATH="/hvs"

# Instalar dependencias
RUN pip install -r requirements-gcp.txt
RUN pip install -r requirements.txt

# Establecer el script que se ejecutará al iniciar el contenedor
ENTRYPOINT ["python", "/hvs/download_data.py"]
