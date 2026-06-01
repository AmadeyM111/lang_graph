class TaskClassification(BaseModel):
    task_type: Literal["code", "dialog", "local"] = Field(
        description="Тип задачи: code - программирование, dialog - общение, local - российские реалии"     
    )

    confidence: float = Field(
        description="Уверенность в классификации от 0.0 до 1.0",
        ge=0.0, le=1.0
    )

    reasoning: str = Field(
        description="Краткое обяснение выбора",
        max_lenght=100
    )