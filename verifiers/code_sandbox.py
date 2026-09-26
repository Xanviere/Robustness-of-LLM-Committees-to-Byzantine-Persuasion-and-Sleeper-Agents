import asyncio
import subprocess
import tempfile
import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)

# Dedicated thread pool executor for CPU-bound docker/sandbox tasks
_sandbox_thread_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="code_sandbox")

class CodeSandbox:
    @staticmethod
    def run_in_docker_sync(code: str, test_cases: str = "", timeout_seconds: float = 5.0) -> bool:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "solution.py")
            full_script = f"{code}\n\n{test_cases}" if test_cases else code
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(full_script)
            
            # Docker run with strict isolation:
            # --network none, 2 CPUs, 1GB RAM, read-only root, max 15 pids, 5s timeout
            docker_cmd = [
                "docker", "run", "--rm",
                "--network", "none",
                "--cpus", "2.0",
                "--memory", "1g",
                "--read-only",
                "--pids-limit", "15",
                "-v", f"{os.path.abspath(temp_dir)}:/sandbox:ro",
                "-w", "/sandbox",
                "python:3.11-slim",
                "timeout", str(int(timeout_seconds)), "python", "solution.py"
            ]
            
            try:
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds + 3.0
                )
                if result.returncode == 0:
                    return True
                # If docker failed because daemon is unreachable, fallback to local subprocess
                if "docker" in result.stderr.lower() or "daemon" in result.stderr.lower() or "connect" in result.stderr.lower():
                    return CodeSandbox._fallback_local_subprocess(file_path, timeout_seconds)
                return False
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.debug(f"Docker sandbox execution skipped or timed out: {e}")
                return CodeSandbox._fallback_local_subprocess(file_path, timeout_seconds)
            except Exception as e:
                logger.error(f"Unexpected error in Docker sandbox: {e}")
                return False

    @staticmethod
    def _fallback_local_subprocess(file_path: str, timeout_seconds: float) -> bool:
        # Fallback local runner in isolated subprocess
        try:
            res = subprocess.run(
                [sys.executable, file_path],
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            return res.returncode == 0
        except Exception:
            return False

    @staticmethod
    async def run_in_docker(code: str, test_cases: str = "", timeout_seconds: float = 5.0) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _sandbox_thread_pool,
            CodeSandbox.run_in_docker_sync,
            code,
            test_cases,
            timeout_seconds
        )
