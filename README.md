# Detecting Intermediate Mass Black Holes in Globular Clusters using Gaia

Este repositorio contiene el código y datos utilizados en el Trabajo de Fin de Máster (TFM) titulado **"Detecting Intermediate Mass Black Holes (IMBH) in globular clusters using Gaia"**. El objetivo de este proyecto es desarrollar un modelo que permita detectar Agujeros Negros de Masa Intermedia en cúmulos globulares utilizando datos astrométricos de la misión Gaia y datos de observatorios de rayos X.

## Descripción del Proyecto

La investigación se enfoca en la identificación de Agujeros Negros de Masa Intermedia (IMBH) en cúmulos globulares, analizando propiedades dinámicas y astrofísicas. Existen dos fuentes de datos principales:
1. **Datos astrométricos de Gaia**: Los datos de Gaia nos permitirán buscar estrellas de hipervelocidad que hayan sido expulsadas de cúmulos globulares densos que alberguen un IMBH.
2. **Datos de rayos X**: La presencia de un IMBH se complementa con la observación de emisiones de rayos X altamente variables y de alta intensidad.

## Objetivos

Este proyecto tiene tres objetivos principales:
1. **Desarrollar un modelo para la detección de IMBH** en cúmulos globulares a partir de los datos de Gaia, utilizando un catálogo de cúmulos globulares.
2. **Publicar los resultados de la búsqueda** de candidatos a cúmulos globulares que puedan albergar un IMBH.
3. Generar un análisis completo de la presencia de emisiones de rayos X en los candidatos identificados.

## Contenido del Repositorio

- `notebooks/`: Jupyter Notebooks para el procesamiento y análisis de datos.
- `data/`: Datos relevantes, incluyendo catálogos de cúmulos globulares y datos de Gaia filtrados.
- `scripts/`: Scripts en Python para el preprocesamiento de datos y cálculos astrofísicos.
- `results/`: Resultados de los análisis, incluyendo posibles candidatos de cúmulos globulares con IMBH.
  
## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/TFM-Detecting-IMBHs.git
   ```
2. Navega al directorio del proyecto:
   ```bash
   cd TFM-Detecting-IMBHs
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Para ejecutar los análisis, abre los notebooks en la carpeta `notebooks/` y sigue las instrucciones paso a paso. Los datos de Gaia se procesan en el notebook principal, y los scripts en `scripts/` se pueden usar para tareas de filtrado y análisis adicionales.

## Contribuciones

Contribuciones y sugerencias son bienvenidas. Si deseas contribuir, realiza un fork del repositorio y abre un Pull Request con tus propuestas.

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Para más detalles, consulta el archivo [LICENSE](LICENSE).
