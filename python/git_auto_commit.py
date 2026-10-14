"""
Ejecuta git add + commit + push automáticamente.

Se ejecuta al final del pipeline para versionar los cambios generados
(nuevos modelos, metadata, scripts actualizados). Si algo falla (sin
cambios, sin conexión a GitHub, etc.), NUNCA rompe el pipeline: siempre
devuelve 0, solo loggea el problema para que quede en el log.

Seguridad: nunca queda commiteado un archivo sensible (.env y similares),
incluso si el .gitignore del repo estuviera mal configurado. No se
usa pathspec negativo en el `git add` -esa combinación tiene un bug real
de git: si el archivo ya está en .gitignore, "git add . ':!.env'" falla
por completo y no agrega NADA, ni siquiera lo seguro-. En su lugar: se
agrega todo normal y luego se revisa qué quedó en stage, sacando (reset)
cualquier archivo sensible que se haya colado (defensa en profundidad,
no depende solo del .gitignore).

"""
from __future__ import annotations

import fnmatch
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("churn.git")

# Directorio raíz del proyecto (el archivo está en python/, así que sube un nivel)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Nunca quedan commiteados archivos que calcen con estos patrones (se
# comparan contra el nombre de archivo y contra la ruta completa).
# Patrones de archivos sensibles. Nota: .env.example/.sample/.template
# SÍ van al repo (son plantillas sin credenciales), por eso se excluyen.
SENSITIVE_PATTERNS = [
    ".env",
    ".env.local",
    ".env.*.local",
    ".env.production",
    ".env.staging",
    ".env.development",
    "*.pem",
    "*.key",
]

# Nombres de archivos que parecen .env pero NO son sensibles (plantillas)
SAFE_ENV_NAMES = [".env.example", ".env.sample", ".env.template"]


def _is_sensitive(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    # Primero: ¿es un archivo seguro explícitamente?
    if any(fnmatch.fnmatch(name, safe) for safe in SAFE_ENV_NAMES):
        return False
    # Segundo: ¿matchea algún patrón sensible?
    return any(
        fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(path, pat)
        for pat in SENSITIVE_PATTERNS
    )


def run_git(args: list[str]) -> tuple[int, str, str]:
    """Ejecuta un comando git y devuelve (returncode, stdout, stderr)."""
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def has_changes() -> bool:
    """¿Hay algo para commitear?"""
    code, output, _ = run_git(["status", "--porcelain"])
    return code == 0 and bool(output.strip())


def env_already_tracked() -> bool:
    """
    ¿.env (u otro archivo sensible) ya quedó commiteado en el pasado?
    Si es así, ya está en el historial de git -borrarlo ahora no alcanza-
    y hace falta rotar credenciales y limpiar el historial a mano.
    """
    for pattern in SENSITIVE_PATTERNS:
        code, out, _ = run_git(["ls-files", "--", pattern])
        if code == 0 and out.strip():
            logger.critical(
                "'%s' ya está versionado en git (archivo: %s). Esto es una fuga de "
                "credenciales. Rota las contraseñas/llaves YA y limpia el historial "
                "(git filter-repo o BFG Repo-Cleaner) antes de seguir usando este repo.",
                pattern, out.strip(),
            )
            return True
    return False


def git_add_safe() -> tuple[int, str, str]:
    """
    git add . normal, y luego revisa qué quedó en stage: si algún archivo
    sensible se coló (ej. .gitignore mal configurado), lo saca del stage
    antes de seguir. No usa pathspec negativo en el add (ver nota arriba).
    """
    code, out, err = run_git(["add", "."])
    if code != 0:
        return code, out, err

    _, staged, _ = run_git(["diff", "--cached", "--name-only"])
    sensible = [f for f in staged.splitlines() if f and _is_sensitive(f)]
    if sensible:
        logger.critical(
            "Se intentó agregar %d archivo(s) sensible(s) al commit: %s. "
            "Se excluyen automáticamente (revisa tu .gitignore).",
            len(sensible), sensible,
        )
        run_git(["reset", "HEAD", "--", *sensible])
    return code, out, err


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Este script es de "mejor esfuerzo": un fallo aquí jamás debe tumbar el
    # pipeline de datos/modelo, que es lo que de verdad importa. Por eso todo
    # el cuerpo va en un try que siempre termina devolviendo 0, salvo la
    # fuga de credenciales, que si merece detener todo.
    try:
        if env_already_tracked():
            return 1  # el único caso que SÍ debe tratarse como fallo real

        if not has_changes():
            logger.info("No hay cambios para versionar. Nada que hacer.")
            return 0

        logger.info("Cambios detectados. Preparando commit...")

        code, _, err = git_add_safe()
        if code != 0:
            logger.error("git add falló (no se bloquea el pipeline): %s", err)
            return 0

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        msg = f"chore(auto): pipeline run at {timestamp}"
        code, out, err = run_git(["commit", "-m", msg])
        if code != 0:
            logger.error("git commit falló (no se bloquea el pipeline): %s | %s", out, err)
            return 0
        logger.info("Commit realizado: %s", msg)

        code, out, err = run_git(["push"])
        if code != 0:
            logger.error(
                "git push falló (no se bloquea el pipeline; el commit quedó "
                "local, se incluirá en el próximo push exitoso): %s | %s", out, err,
            )
            return 0
        logger.info("Push a origin completado")

        return 0
    except Exception:
        logger.exception("Falló el auto-commit (no se bloquea el pipeline)")
        return 0


if __name__ == "__main__":
    sys.exit(main())