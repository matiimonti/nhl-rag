"""
Inter-container networking test.
Verifies every container can reach its dependencies by Docker service name.

Run from host:  python tests/test_container_networking.py
Requires: Docker running with all containers up.
"""
import subprocess
import sys

# (container, target_host, target_port)
CHECKS = [
    ("nhlrag_ingestion", "qdrant",  6333),
    ("nhlrag_ingestion", "redis",   6379),
    ("nhlrag_api",       "qdrant",  6333),
    ("nhlrag_api",       "redis",   6379),
    ("nhlrag_streamlit", "fastapi", 8000),
]

PROBE = (
    "import socket, sys; s = socket.create_connection(('{host}', {port}), timeout=3); "
    "s.close(); print('OK')"
)


def check(container: str, host: str, port: int) -> bool:
    cmd = [
        "docker", "exec", container,
        "python3", "-c", PROBE.format(host=host, port=port),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        ok = result.returncode == 0 and "OK" in result.stdout
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}]  {container}  →  {host}:{port}")
        if not ok:
            print(f"         stderr: {result.stderr.strip()}")
        return ok
    except subprocess.TimeoutExpired:
        print(f"  [FAIL]  {container}  →  {host}:{port}  (timeout)")
        return False
    except FileNotFoundError:
        print("ERROR: 'docker' not found in PATH")
        sys.exit(1)


if __name__ == "__main__":
    print("Container-to-container network checks\n")
    results = [check(c, h, p) for c, h, p in CHECKS]
    print()
    if all(results):
        print("All checks passed.")
        sys.exit(0)
    else:
        failed = sum(1 for r in results if not r)
        print(f"{failed} check(s) FAILED.")
        sys.exit(1)
