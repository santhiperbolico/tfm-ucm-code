from os import path

import setuptools

# Obtener la ruta raíz del proyecto
root_dir = path.abspath(path.dirname(__file__))

# Leer las dependencias desde requirements.txt
requirements_path = path.join(root_dir, "requirements.txt")
with open(requirements_path, "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="hyper_velocity_stars_detection",
    version="0.0.1",
    author="Santiago Arranz",
    author_email="santiago.arranz.sanz@gmail.com",
    description="Librería para la detección de HVS.",
    long_description_content_type="text/markdown",
    url="https://github.com/santhiperbolico/tfm-ucm-code",
    install_requires=requirements,
    packages=setuptools.find_packages("src"),  # Busca paquetes en 'src'
    package_dir={"": "src"},  # Indica que los paquetes están en 'src'
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
