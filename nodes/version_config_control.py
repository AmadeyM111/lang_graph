import time
from typing import Any

class ModelConfig:
    def __init__(self):
        self.models = {
            "classifier": deepseek_model,
            "coder": deepseek_model,
            "dialog": amvera_model,
            "local": gigachat_model,
            "synthesizer": amvera_model
        }

    def get_model(self, role: str):
        """Получить модель по роли с возможность А/В тестирования"""
        if role in self.models:
            return self.models[role]
        return self.models["dialog"] # fallback

    def switch_model(self, role: str, new_model):
        """Горячая замена модели"""
        self.models[role] = new_model

# ------------ Мониторинг работы модели -------------------

def monitor_model_performance(state: MultiModelState) -> dict:
    """ Отслеживание производительности моделей """
    start_time = state.get("start_time")

    metrics = {
        "classification_confidence": state.get("classification_confidence", 0),
        "response_time": time.time() - start_time if start_time else None,
        "model_used": state.get("task_type", "unknown"),
        "success": bool(state.get("final_answer"))
    }

    # Логирование метрик
    log_metrics(metrics)

    return {"metrics": metrics}
