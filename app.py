import uvicorn
from app.api.endpoints import app
if __name__ == "__main__":


    uvicorn.run("app.api.endpoints:app",
                host="127.0.0.1",
                port=9000,
                reload=True,
                log_level="info")