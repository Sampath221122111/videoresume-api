from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.config import get_settings
import httpx

security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    token = credentials.credentials
    settings = get_settings()

    try:
        resp = httpx.get(
            f"{settings.supabase_url}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": settings.supabase_key,
                "Accept": "application/json",
            },
            timeout=15.0,
        )
        print(f"[AUTH] Supabase response: {resp.status_code}")

        if resp.status_code == 200:
            user_data = resp.json()
            user_id = user_data.get("id", "")
            if user_id:
                print(f"[AUTH] Verified user: {user_id}")
                return {
                    "user_id": user_id,
                    "email": user_data.get("email", ""),
                    "payload": user_data,
                }

        print(f"[AUTH] Failed: {resp.text[:200]}")
    except Exception as e:
        print(f"[AUTH] Error: {e}")

    raise HTTPException(status_code=401, detail="Authentication failed")
