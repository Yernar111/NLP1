import torch
from typing import List, Tuple
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import settings
from api.schemas import SourceDocument
from src.prompt import SYSTEM_PROMPT

class RAGService:
    def __init__(self):
        print(f"Инициализация RAGService на устройстве: {settings.DEVICE}...")

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.qdrant_client = QdrantClient(path=settings.QDRANT_PATH)
        
        self.embedder = SentenceTransformer(settings.EMBEDDER_MODEL_NAME, device=settings.DEVICE)
        
        self.tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.BASE_MODEL_NAME,
            torch_dtype=torch_dtype,
            device_map=settings.DEVICE
        )

    def retrieve_context(self, query: str, top_k: int) -> Tuple[str, List[SourceDocument]]:
        query_vector = self.embedder.encode(query).tolist()
        
        search_results = self.qdrant_client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        ).points

        sources = []
        retrieved_texts = []

        for hit in search_results:
            text = hit.payload.get("text", "")
            passage_id = hit.payload.get("passage_id", hit.id)
            
            sources.append(SourceDocument(
                passage_id=passage_id,
                score=hit.score,
                text=text
            ))
            retrieved_texts.append(text)

        context_block = "\n---\n".join(retrieved_texts)
        return context_block, sources

    def generate_answer(self, query: str, context: str) -> str:

        user_content = f"Context:\n{context}\n\nQuery: {query}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(settings.DEVICE)

        # Инференс модели с жадным декодированием (greedy search) для воспроизводимости
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                temperature=settings.TEMPERATURE,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        
        answer = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return answer.strip()

    def answer_query(self, query: str, top_k: int) -> Tuple[str, List[SourceDocument]]:
        """ Полный RAG-пайплайн """
        context, sources = self.retrieve_context(query, top_k)
        
        if not context:
            return "Информация в базе знаний не найдена.", []
            
        response = self.generate_answer(query, context)
        return response, sources