# Use uma imagem base oficial do Python.
# A versão 'slim-buster' é menor e contém apenas o essencial, o que é bom para produção.
FROM python:3.9-slim-buster

# Define o diretório de trabalho dentro do contêiner.
# Todos os comandos subsequentes serão executados a partir deste diretório.
WORKDIR /app

# Copia o arquivo requirements.txt para o diretório de trabalho do contêiner.
# É uma boa prática copiar apenas o requirements.txt primeiro para aproveitar o cache do Docker.
COPY requirements.txt .

# Instala as dependências Python especificadas no requirements.txt.
# O --no-cache-dir evita que o pip armazene pacotes em cache, reduzindo o tamanho da imagem.
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o conteúdo do diretório atual (o seu projeto) para o diretório de trabalho do contêiner (/app).
# Isso inclui src/, notebooks/, e quaisquer outros arquivos.
COPY . .

# Expõe a porta que a sua API FastAPI ou Flask estará escutando.
# Por padrão, FastAPI usa a porta 8000.
EXPOSE 8000

# Comando para iniciar a aplicação quando o contêiner for executado.
# Aqui estamos usando Uvicorn para rodar a aplicação FastAPI.
# - src.main:app: Indica que o objeto 'app' da aplicação FastAPI está no arquivo 'main.py' dentro do diretório 'src'.
# - --host 0.0.0.0: Faz com que a API esteja acessível de qualquer interface de rede dentro do contêiner,
#                     o que é necessário para acessá-la de fora do contêiner.
# - --port 8000: Especifica a porta em que a API vai escutar.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]