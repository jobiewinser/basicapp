- To add a python package:
docker run --rm -v "${PWD}:/app" -w /app python:3.10-slim sh -c "cd backend && pip install poetry && poetry add {package} && poetry lock"

- To add an npm package:
cd frontend/my-app
rm package.lock.json
rm nodemodules (this might make it take a long time, maybe optional?)
npm install {package} --save

- To start the whole project while devving
docker-compose -f docker-compose.dev.yml up --build

- To debug mode in python
press f5 in vscode

- Reach frontend while devving:
start server then http://127.0.0.1:3000/

structure as of 18:45 07/02/2025
BasicApp
 ┣ frontend
 ┃ ┣ my-app
 ┃ ┃ ┣ public
 ┃ ┃ ┃ ┣ favicon.ico
 ┃ ┃ ┃ ┣ index.html
 ┃ ┃ ┃ ┣ logo192.png
 ┃ ┃ ┃ ┣ logo512.png
 ┃ ┃ ┃ ┣ manifest.json
 ┃ ┃ ┃ ┗ robots.txt
 ┃ ┃ ┣ src
 ┃ ┃ ┃ ┣ App.css
 ┃ ┃ ┃ ┣ App.js
 ┃ ┃ ┃ ┣ App.test.js
 ┃ ┃ ┃ ┣ index.css
 ┃ ┃ ┃ ┣ index.js
 ┃ ┃ ┃ ┣ logo.svg
 ┃ ┃ ┃ ┣ reportWebVitals.js
 ┃ ┃ ┃ ┗ setupTests.js
 ┃ ┃ ┣ .gitignore
 ┃ ┃ ┣ package-lock.json
 ┃ ┃ ┣ package.json
 ┃ ┃ ┗ README.md
 ┃ ┣ Dockerfile.dev.react
 ┃ ┣ Dockerfile.prod.react
 ┃ ┣ package-lock.json
 ┃ ┗ package.json
 ┣ .env
 ┣ .gitignore
 ┣ docker-compose.dev.yml
 ┣ docker-compose.prod.yml
 ┣ nginx.dev.conf
 ┣ nginx.prod.conf
 ┗ README.md