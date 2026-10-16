from api.services import RAGService

# Глобальный объект сервиса (состояние синглтона)
rag_service_instance: RAGService | None = None

def get_rag_service() -> RAGService:
    """ Dependency Provider для внедрения сервиса в FastAPI эндпоинты """
    if rag_service_instance is None:
        raise RuntimeError("RAGService еще не инициализирован!")
    return rag_service_instance