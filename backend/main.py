import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI
from langfuse import Langfuse

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# --------------------------
# LiteLLM Proxy Client
# -------------------------

client = OpenAI(
    api_key = os.getenv("LITELLM_API_KEY"),
    base_url = os.getenv("LITELLM_BASE_URL"),
)


# --------------------------
# Langfuse Cloud
# --------------------------
langfuse = Langfuse(
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key = os.getenv("LANGFUSE_SECRET_KEY"),
    host = os.getenv("LANGFUSE_HOST")
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    with langfuse.start_as_current_observation(
        name="fastapi-chat",
        as_type="span",
        input={"message": request.message},
    ) as trace:
        try:
            with langfuse.start_as_current_observation(
                name="litellm-generation",
                as_type="generation",
                model="gpt4",
                input=request.message,
            ) as generation:
                response = client.chat.completions.create(
                    model="gpt4",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful AI assistant.",
                        },
                        {
                            "role": "user",
                            "content": request.message,
                        },
                    ],
                )

                answer = response.choices[0].message.content

                generation.update(
                    output=answer,
                    usage_details={
                        "input": response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens,
                        "total": response.usage.total_tokens,
                    },
                )

            trace.update(output={"answer": answer})
            langfuse.flush()

            return {"answer": answer}

        except Exception as error:
            trace.update(
                level="ERROR",
                status_message=str(error),
                output={"error": str(error)},
            )
            langfuse.flush()
            raise
