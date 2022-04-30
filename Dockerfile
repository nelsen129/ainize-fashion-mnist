FROM pnelsen129/fashion-mnist

WORKDIR /app

RUN pip install flask tensorflow Pillow

COPY . /app

CMD ["python3", "app.py"]
