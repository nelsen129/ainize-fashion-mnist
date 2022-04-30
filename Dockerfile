FROM pnelsen129/fashion-mnist:1

WORKDIR /app

RUN pip install flask tensorflow Pillow

COPY . /app
COPY ./frontend /app/frontend

CMD ["python3", "app.py"]
