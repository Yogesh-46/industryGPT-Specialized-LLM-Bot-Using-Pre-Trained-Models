"""robots.txt checking helpers."""

from __future__ import annotations

import logging
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

logger = logging.getLogger(__name__)


class RobotsCache:
    """Cache RobotFileParser instances per origin."""

    def __init__(self, user_agent: str, timeout: float = 15.0) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._parsers: dict[str, RobotFileParser | None] = {}

    def _origin(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def get_parser(self, url: str) -> RobotFileParser | None:
        origin = self._origin(url)
        if origin in self._parsers:
            return self._parsers[origin]

        robots_url = f"{origin}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            # RobotFileParser.read() uses urllib without timeout on some versions;
            # fetch via parser API which is acceptable for academic polite collection.
            parser.read()
            self._parsers[origin] = parser
            logger.info("Loaded robots.txt for %s", origin)
            return parser
        except Exception as exc:  # noqa: BLE001 - network/robots failures are expected
            # Fail open with a warning would be risky; fail closed for unknown robots.
            # Academic requirement: respect robots.txt. If unreadable, skip URL.
            logger.warning(
                "Could not read robots.txt at %s (%s). Treating as disallow.",
                robots_url,
                exc,
            )
            self._parsers[origin] = None
            return None

    def is_allowed(self, url: str) -> bool:
        """Return True if user-agent may fetch ``url`` per robots.txt."""
        parser = self.get_parser(url)
        if parser is None:
            return False
        try:
            return bool(parser.can_fetch(self.user_agent, url))
        except Exception as exc:  # noqa: BLE001
            logger.warning("robots check failed for %s: %s", url, exc)
            return False
