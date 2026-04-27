"""
Centralized configuration — loads from .env.
NO CACHING — always reads fresh values.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    supabase_url: str
    supabase_key: str
    supabase_service_key: str
    supabase_jwt_secret: str
    cloudinary_cloud_name: str
    cloudinary_api_key: str
    cloudinary_api_secret: str
    groq_api_key: str
    upstash_redis_url: str

    frontend_url: str = "http://localhost:3000"
    max_video_size_mb: int = 200
    max_video_duration_sec: int = 300
    max_submissions_per_user: int = 1000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    return Settings()
