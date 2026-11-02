"""Configurações do projeto (variáveis de ambiente / .env)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_path: str = "data/raw/ktas_triagem_ps.csv"
    model_path: str = "models/model.joblib"
    seed: int = 42


settings = Settings()
