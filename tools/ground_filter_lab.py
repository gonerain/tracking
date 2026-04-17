from __future__ import annotations

from _bootstrap import bootstrap


def run() -> None:
    from core.ground_filter_lab import main

    main()


if __name__ == "__main__":
    bootstrap(run)
