FROM pnelsen129/fashion-mnist

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

CMD ["python3", "app.py"]
