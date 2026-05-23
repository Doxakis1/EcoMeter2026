from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json

app = FastAPI()

class Item(BaseModel):
    title: str
    description: str | None = None
    price: float

@app.get("/")
def read_root():
    return {"status": "online", "message": "Καλώς ήρθες στο API σου!"}

@app.post("/items/")
def create_item(item: Item):
    return {"message": "Το αντικείμενο αποθηκεύτηκε", "data": item}

# --- ENDPOINT ΜΕ ΤΥΠΩΜΑ ΑΠΑΝΤΗΣΗΣ ΣΤΟ ΤΕΡΜΑΤΙΚΟ ---
@app.get("/prompt")
def receive_prompt(user_input: str):
    # 1. Τύπωμα του prompt που ήρθε από τον χρήστη
    print(f"\n>>> [NEW PROMPT] user entered prompt: {user_input}")
    
    bot_response = f"Server with port 5542 received the prompt: '{user_input}'."
    
    # 2. Αποθηκεύουμε την απάντηση σε ένα dictionary (μεταβλητή)
    response_data = {
        "status": "success",
        "user_prompt": user_input,
        "server_response": bot_response
    }
    
    # 3. Μετατρέπουμε το dictionary σε όμορφο string και το τυπώνουμε στο τερματικό
    print(">>> [SERVER RESPONSE] Ο server responds with JSON:")
    print(json.dumps(response_data, indent=4, ensure_ascii=False))
    print("-" * 60) 
    
    # 4. Στέλνουμε την απάντηση πίσω στον browser
    return response_data

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=5542, reload=True)