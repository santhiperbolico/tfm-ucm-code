import os

from google.cloud import storage


def upload_folder_to_gcs(project_id, bucket_name, temp_path, destination_folder):
    """Sube todos los archivos de temp_path a una carpeta en Google Cloud Storage"""
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    # Recorrer todos los archivos en la carpeta temporal
    for file_name in os.listdir(temp_path):
        file_path = os.path.join(temp_path, file_name)

        # Verificar que sea un archivo y no un directorio
        if os.path.isfile(file_path):
            # Definir la ruta en el bucket
            blob_path = f"{destination_folder}/{file_name}"
            blob = bucket.blob(blob_path)
            # Subir el archivo
            blob.upload_from_filename(file_path)
