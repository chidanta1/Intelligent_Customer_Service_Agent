"""Build the isolated GraphRAG index used by Experiment 3.

This wrapper reuses the backend's DeepSeek credential in memory and never
duplicates it into ``app/graphrag/data/.env``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2]
GRAPHRAG_PACKAGE_ROOT = BACKEND_ROOT / "app" / "graphrag"
GRAPHRAG_DATA_ROOT = GRAPHRAG_PACKAGE_ROOT / "data"
sys.path.insert(0, str(GRAPHRAG_PACKAGE_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import settings  # noqa: E402
from graphrag.cli.index import index_cli  # noqa: E402
from graphrag.config.enums import IndexingMethod  # noqa: E402
from graphrag.logger.types import LoggerType  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-connectivity-validation", action="store_true")
    args = parser.parse_args()

    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is required to build the GraphRAG index")
    os.environ["GRAPHRAG_API_KEY"] = settings.DEEPSEEK_API_KEY
    index_cli(
        root_dir=GRAPHRAG_DATA_ROOT,
        method=IndexingMethod.Standard,
        verbose=True,
        memprofile=False,
        cache=True,
        logger=LoggerType.PRINT,
        config_filepath=None,
        dry_run=args.dry_run,
        skip_validation=args.skip_connectivity_validation,
        output_dir=None,
    )


if __name__ == "__main__":
    main()
