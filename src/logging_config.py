import logging


def get_logger(trace_id: str = "default"):
    logger = logging.getLogger(f"self_correcting_rag.{trace_id}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return _BoundLogger(logger)


class _BoundLogger:
    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def _format(self, event: str, **kwargs) -> str:
        fields = " ".join(f"{k}={v}" for k, v in kwargs.items())
        return f"{event} {fields}".strip()

    def info(self, event: str, **kwargs) -> None:
        self._logger.info(self._format(event, **kwargs))

    def warning(self, event: str, **kwargs) -> None:
        self._logger.warning(self._format(event, **kwargs))

    def debug(self, event: str, **kwargs) -> None:
        self._logger.debug(self._format(event, **kwargs))

    def error(self, event: str, **kwargs) -> None:
        self._logger.error(self._format(event, **kwargs))

    def exception(self, event: str, **kwargs) -> None:
        self._logger.error(self._format(event, **kwargs), exc_info=True)