FROM pnelsen129/fashion-mnist:1

WORKDIR /app

RUN pip install flask tensorflow Pillow

COPY . /app

CMD ["python3", "app.py"]
