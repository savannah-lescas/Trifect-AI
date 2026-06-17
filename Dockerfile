FROM public.ecr.aws/docker/library/python:3.11-slim-bookworm

COPY . /trivox
WORKDIR /trivox

RUN pip3 install --no-cache-dir \
    Flask \
    flask-cors \
    oci \
    authlib \
    requests \
    pypdf

EXPOSE 8000
CMD ["python3", "./trivox.py"]
