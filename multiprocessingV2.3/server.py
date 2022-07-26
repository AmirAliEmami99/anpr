from fastapi import FastAPI
app = FastAPI(title='Hello world')

@app.get('/index')
async def hello_world():
    return "hello world"