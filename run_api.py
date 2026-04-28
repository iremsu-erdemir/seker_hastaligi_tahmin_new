"""FastAPI servisini yerelden hızlıca başlatmak için giriş script'i."""

from __future__ import annotations

import uvicorn


if __name__ == "__main__":
    # Varsayılan olarak localhost:8000 üzerinde API ayağa kaldırılır.
    uvicorn.run("src.diabetes_adaboost.api:app", host="0.0.0.0", port=8000, reload=False)

