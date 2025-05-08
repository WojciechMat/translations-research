import os
from time import sleep
from typing import Generator

import dotenv
from google import genai
from google.genai import types
from omegaconf import DictConfig

dotenv.load_dotenv()


class GeminiClient:
    def __init__(self, generation_cfg: DictConfig):
        try:
            gen_api_key = os.environ.get("GENAI_API_KEY")
        except KeyError:
            raise KeyError("Add GENAI_API_KEY to your .env before using GeminiClient")
        self.client = genai.Client(api_key=gen_api_key)
        self.max_output_tokens = generation_cfg["max_new_tokens"]
        self.model_path = generation_cfg["model_path"]
        self.temperature = generation_cfg["temperature"]
        self.n = 0

    def message(self, prompt: str) -> Generator[str, str, str]:
        # NOTE: This is not a chat. This will send only one instruction. Chat is not implemented yet.
        response = self.client.models.generate_content_stream(
            model=self.model_path,
            contents=[prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
            ),
        )
        for chunk in response:
            yield chunk.text

    def prompt(self, prompt: str) -> str:
        # Try to call whatever free model is available (useful for long lightrag indexing sessions)
        self.n += 1
        for retry in range(10):
            try:
                response = self.client.models.generate_content(
                    model=self.model_path,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        max_output_tokens=self.max_output_tokens,
                        temperature=self.temperature,
                    ),
                )
                break
            except Exception as e:
                print(f"Exception prompting Gemini, trying to prompt 2.0-flash-lite next: {e}")
                try:
                    response = self.client.models.generate_content(
                        model="gemini-2.0-flash-lite",
                        contents=[prompt],
                        config=types.GenerateContentConfig(
                            max_output_tokens=self.max_output_tokens,
                            temperature=self.temperature,
                        ),
                    )
                    break
                except Exception as e:
                    print(f"Exception prompting Gemini, trying to prompt 2.0-flash next: {e}")
                    try:
                        response = self.client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=[prompt],
                            config=types.GenerateContentConfig(
                                max_output_tokens=self.max_output_tokens,
                                temperature=self.temperature,
                            ),
                        )
                        break
                    except Exception as e:
                        print(f"Exception prompting Gemini, sleeping for 10 seconds: {e}")
                        sleep(10)
        print(f"Call no. {self.n} done")
        return response.text

    def get_conversation_context(self) -> list[str]:
        # No chat - no context
        return []

    def reset_chat(self):
        # No chat to reinitialize: do nothing
        pass
