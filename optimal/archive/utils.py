import logging

def setup_strict_logging(_level=logging.DEBUG):
    """Call this once at the start of your program to enforce strict logging everywhere."""
    logging.basicConfig(
        level=_level,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True  # 'force=True' overrides any other logging setup from third-party libraries
    )