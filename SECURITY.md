# Seguridad

Controles aplicados al prototipo (CLI y API):
- **Validación de entradas:** tipo `str`, no vacío, máx. 1.000 caracteres y máx. 64 textos por petición (HTTP 422 si no se cumple).
- **Sin deserialización insegura:** el vocabulario se guarda en JSON (no `pickle`); el modelo en formato `.keras`. Cargue solo modelos de fuentes confiables.
- **Privacidad:** la API no registra los textos de los usuarios; el registro de la CLI es opcional (`--registro`).
- **Contenedor sin privilegios:** el `Dockerfile` ejecuta como usuario no root, con dependencias fijadas en `requirements.txt` y sin secretos en el repositorio.
- **Pendiente para producción:** autenticación, límites de tasa (rate limiting), HTTPS en el proxy inverso y monitoreo de deriva.

Reporte de vulnerabilidades: abra un *issue* privado o escriba a los autores.
