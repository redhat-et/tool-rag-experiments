import contextlib
import gc
import logging
import socket
import threading
import time
from typing import Optional

import uvicorn


class UvicornServerSwitcher:
    """
    Minimal manager that runs a given ASGI app (e.g., from mcp.streamable_http_app())
    on a background Uvicorn thread. Calling `switch(new_app)` gracefully stops the
    current server (if any), waits for the port to be freed, and starts the new one.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 3333,
        graceful_shutdown: int = 15,
        post_stop_port_wait: float = 10.0,
        readiness_probe: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.graceful_shutdown = graceful_shutdown
        self.post_stop_port_wait = post_stop_port_wait
        self.readiness_probe = readiness_probe
        self.log = logger or logging.getLogger("MCPServerSwitcher")

        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    # ---------- public API ----------

    def switch(self, app) -> None:
        """
        Start serving `app` on (host, port), replacing any existing server.
        """
        with self._lock:
            self._stop_unlocked()
            self._start_unlocked(app)

    def stop(self) -> None:
        """
        Stop the server if running.
        """
        with self._lock:
            self._stop_unlocked()

    def is_running(self) -> bool:
        t = self._thread
        return bool(t and t.is_alive())

    # ---------- internals ----------

    def _start_unlocked(self, app) -> None:
        self.log.info("Starting MCP server on %s:%d ...", self.host, self.port)

        config = uvicorn.Config(
            app=app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=False,
            workers=1,                       # single worker for clean in-proc control
            lifespan="on",                   # run startup/shutdown hooks
            timeout_graceful_shutdown=self.graceful_shutdown,
        )
        server = uvicorn.Server(config)

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        self._server = server
        self._thread = thread

        if self.readiness_probe:
            self._wait_for_listening()

        self.log.info("MCP server is ready on %s:%d", self.host, self.port)

    def _stop_unlocked(self) -> None:
        if not self._server or not self._thread:
            return

        self.log.info("Stopping MCP server on %s:%d ...", self.host, self.port)
        # Ask Uvicorn to exit gracefully.
        self._server.should_exit = True

        # Wait for the serving thread to finish (graceful + small buffer).
        join_timeout = self.graceful_shutdown + 5.0
        if self._thread.is_alive():
            self._thread.join(timeout=join_timeout)

        # Clear references so Python can reclaim memory (important with many tools).
        self._server = None
        self._thread = None

        # Wait for the OS to actually release the port (handles TIME_WAIT).
        self._wait_for_port_free()

        # Encourage cleanup of large graphs/schemas.
        gc.collect()

        self.log.info("MCP server stopped and port freed.")

    # ---------- tiny helpers ----------

    def _wait_for_listening(self, timeout: float = 10.0, tick: float = 0.05) -> None:
        deadline = time.time() + timeout
        addr = ("127.0.0.1", self.port)  # connect locally to confirm accept()
        while time.time() < deadline:
            with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.settimeout(tick)
                try:
                    if s.connect_ex(addr) == 0:
                        return
                except OSError:
                    pass
            time.sleep(tick)
        raise TimeoutError(f"Server on {self.host}:{self.port} did not start within {timeout:.1f}s")

    def _wait_for_port_free(self, timeout: float = None, tick: float = 0.1) -> None:
        timeout = self.post_stop_port_wait if timeout is None else timeout
        deadline = time.time() + timeout
        while time.time() < deadline:
            with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind((self.host, self.port))
                    return
                except OSError:
                    time.sleep(tick)
        raise TimeoutError(f"Port {self.host}:{self.port} did not become free within {timeout:.1f}s")
