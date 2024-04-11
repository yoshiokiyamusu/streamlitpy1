FROM python:3.10

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app

COPY . .

EXPOSE 8000

ENTRYPOINT [ "streamlit", "run"]

CMD ["app.py"]
