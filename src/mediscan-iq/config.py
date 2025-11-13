from typing import List
from pydantic import Field, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ===== Server =====
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: PositiveInt = Field(default=8000, alias="API_PORT")

    # ===== Ingestion constraints =====
    max_chars: PositiveInt = Field(default=20000, alias="MAX_CHARS")
    accepted_report_types: List[str] = Field(
        default_factory=lambda: ["radiology", "pathology", "discharge", "ecg", "echo", "others"],
        alias="ACCEPTED_REPORT_TYPES",
    )

    # ===== Anonymizer config =====
    anonymize_strategy: str = Field(default="hash", alias="ANONYMIZE_STRATEGY")  # mask | hash
    mask_char: str = Field(default="█", alias="MASK_CHAR")
    hash_salt: str = Field(default="mediscan", alias="HASH_SALT")
    reduce_whitespace: bool = Field(default=True, alias="REDUCE_WHITESPACE")
    keep_dates: bool = Field(default=False, alias="KEEP_DATES")

    # ===== Models / Inference =====
    summarizer_model: str = Field(default="google/flan-t5-base", alias="SUMMARIZER_MODEL")
    summarizer_max_input_tokens: int = Field(default=2048, alias="SUMMARIZER_MAX_INPUT_TOKENS")
    summarizer_max_output_tokens: int = Field(default=128, alias="SUMMARIZER_MAX_OUTPUT_TOKENS")
    summarizer_num_beams: int = Field(default=4, alias="SUMMARIZER_NUM_BEAMS")
    summarizer_temperature: float = Field(default=0.0, alias="SUMMARIZER_TEMPERATURE")
    summarizer_prompt_style: str = Field(default="radiology_brief", alias="SUMMARIZER_PROMPT_STYLE")

    risk_nli_model: str = Field(default="facebook/bart-large-mnli", alias="RISK_NLI_MODEL")
    risk_labels_csv: str = Field(default="low risk,moderate risk,high risk", alias="RISK_LABELS_CSV")
    risk_threshold_high: float = Field(default=0.64, alias="RISK_THRESHOLD_HIGH")
    risk_threshold_moderate: float = Field(default=0.42, alias="RISK_THRESHOLD_MODERATE")

    # ===== Runtime =====
    device_preference: str = Field(default="auto", alias="DEVICE_PREFERENCE")  # auto|cpu|cuda|mps

    # ===== Heuristics / Domain =====
    risk_heuristics_enabled: bool = Field(default=True, alias="RISK_HEURISTICS_ENABLED")

    # ===== Reproducibility =====
    seed: int = Field(default=42, alias="SEED")

    # Pydantic Settings v2 config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
    )

    # Robustly parse comma-separated env for list fields
    @field_validator("accepted_report_types", mode="before")
    @classmethod
    def _split_types(cls, v):
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return v


settings = Settings()
