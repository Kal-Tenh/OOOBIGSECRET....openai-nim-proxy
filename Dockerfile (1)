FROM node:18-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY package*.json ./
RUN npm install --omit=dev

# Copy the rest of the app
COPY . .

# Fly.io / most PaaS providers set PORT via env var — make sure server.js reads process.env.PORT
EXPOSE 8080

CMD ["node", "server.js"]
