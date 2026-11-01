"""Explicit opt-in NVIDIA smoke check: ``python -m data_synth.smoke --live``."""

import argparse
import json

from sqlalchemy.orm import Session

from .cache import CacheError, ResponseCache
from .config import Settings
from .db import init_db, make_engine
from .model_service import run_model
from .providers import ModelRequest, OpenAIChatProvider, ProviderError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="allow an uncached, billable API call")
    parser.add_argument("--model", help="override NVIDIA_MODEL")
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("--live is required to allow a real NVIDIA request")
    engine = None
    try:
        settings = Settings.load()
        cache = ResponseCache(settings.cache_dir)
        provider = OpenAIChatProvider(
            api_key=settings.nvidia_api_key.get_secret_value() if settings.nvidia_api_key else None,
            cache=cache,
            endpoint=settings.nvidia_base_url,
        )
        request = ModelRequest(
            provider="nvidia",
            endpoint=settings.nvidia_base_url,
            model=args.model or settings.nvidia_model,
            prompt_version="numeric-smoke-v1",
            messages=[
                {"role": "user", "content": "Which number is larger, 9.11 or 9.8? Be brief."}
            ],
            parameters={"temperature": 0, "top_p": 1, "max_tokens": 1024},
        )
        engine = make_engine(settings.database_url)
        init_db(engine)
        with Session(engine) as session:
            run = run_model(session, provider, request)
            session.commit()
            print(
                json.dumps(
                    {
                        "run_id": run.id,
                        "model": run.model,
                        "cache_hit": run.cache_hit,
                        "status": "completed",
                    }
                )
            )
        return 0
    except (ProviderError, CacheError) as error:
        print(str(error))
        return 1
    except Exception:
        # DB URLs and provider exceptions may contain credentials: never echo them.
        print("Smoke check failed during configuration or persistence; inspect local setup.")
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
