FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

ENTRYPOINT ["streamlit", "run", "LLM_Behavior_Tester.py", "--server.port=8501", "--server.address=0.0.0.0"]