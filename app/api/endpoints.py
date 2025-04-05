from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.shop_assist import LaptopShopAssistant
from app.models.schemas import ChatResponse, UserRequest

app = FastAPI(title="Laptop Shop Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/Chat", response_model=ChatResponse)
async def chat(request: UserRequest):
    assistant = LaptopShopAssistant('app/api/laptop_data.csv')
    return await assistant.process_request(request)

