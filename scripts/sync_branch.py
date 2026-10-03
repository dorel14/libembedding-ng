#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
 * libembedding - scripts/sync_branch.py
 * Fusionne une branche (par defaut origin/main) dans la branche courante en
 * resolvant automatiquement les conflits dont la seule cause est un tampon de
 * version, puis en re-synchronisant tous les tampons sur la version publiee.
 *
 * Auteur: David Orel
 * Version: 1.9.0
 *
 * Pourquoi: une publication semantic-release ecrit la version dans ~72 fichiers
 * (les trois sources de verite machines, puis les ~70 en-tetes de documentation
 * via scripts/sync_version.py). Si la branche de developpement tamponne elle
 * aussi une version, git voit la meme ligne modifiee des deux cotes et declare
 * un conflit sur chacun de ces 72 fichiers. Or ces conflits sont triviaux: le
 * contenu utile vient du developpement, seule la version vient de la release.
 *
 * Ce qui decide n'est pas l'apparence du hunk mais une question factuelle:
 * "la branche entrante a-t-elle change autre chose que des numeros de version
 * dans ce fichier depuis notre base commune ?"
 *   - non  -> on garde le contenu du developpement, puis la version est
 *             reecrite par scripts/sync_version.py. Rien de perdu.
 *   - oui  -> abandon explicite, avec la liste des fichiers. Un correctif ou un
 *             changement de contenu venu de main meritait une decision humaine,
 *             pas un --ours silencieux qui l'effacerait.
 *
 * La distinction compte: dans config.h le hunk conflictuel melange les macros
 * de version et une fonction ajoutee par le developpement. Il est resolvable,
 * mais pas parce qu'il "a l'air" version-only: parce que main n'a rien change
 * d'autre dans ce fichier. Le test porte sur le diff main-vs-base, jamais sur
 * la ressemblance du hunk.
 *
 * Usage:
 *   git switch dev && git fetch origin main
 *   python scripts/sync_branch.py --ref origin/main --dry-run   # rapport seul
 *   python scripts/sync_branch.py --ref origin/main             # commit de fusion
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_TOML = "python/pyproject.toml"

#: Fichiers entierement regeneres : inutile de les fusionner ligne a ligne, il
#: suffit de les reconstruire apres coup (cf. generate-llm-txt.yml).
GENERATED = ("llm.txt", "llm_full.txt")

#: Une ligne "tampon" est soit une source machine (LIBEMBEDDING_VERSION, qui
#: couvre set(...), les macros MAJOR/MINOR/PATCH sans point et _STRING), soit un
#: en-tete qui se dit version *et* porte un X.Y.Z, soit la marque (vX.Y.Z) de
#: _cdefs.h. Une prose comme "Versioning is handled by" ne porte pas de X.Y.Z et
#: ne correspond donc pas.
STAMP_LIKE = re.compile(
    r"LIBEMBEDDING_VERSION"
    r"|[Vv]ersion[^\n]*?\d+\.\d+\.\d+"
    r"|\(v\d+\.\d+\.\d+\)",
)


class SyncError(RuntimeError):
    """Erreur actionnable, destinee a etre lue telle quelle dans les logs CI."""


def git(*args: str, check: bool = True) -> str:
    """Execute git et retourne stdout (sans le retour chariot final)."""
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        raise SyncError(f"git {' '.join(args)} a echoue :\n{result.stderr.strip()}")
    return result.stdout


def canonical_version(ref: str) -> str:
    """Version publiee, lue dans python/pyproject.toml du ref donne.

    C'est la source de verite du projet (cf. [tool.semantic_release] version_toml).
    """
    content = git("show", f"{ref}:{CANONICAL_TOML}")
    match = re.search(r'^version = "([^"]+)"', content, re.MULTILINE)
    if not match:
        raise SyncError(f"version introuvable dans {ref}:{CANONICAL_TOML}")
    return match.group(1)


def unmerged_paths() -> list[str]:
    out = git("diff", "--name-only", "--diff-filter=U")
    return [line for line in out.splitlines() if line]


def substantive_changes(base: str, ref: str, path: str) -> list[str]:
    """Lignes non-tampon que *ref* a modifiees dans *path* depuis *base*.

    C'est la seule question qui compte : si la liste est vide, prendre le
    contenu du developpement ne perd rien de la release.
    """
    diff = git("diff", base, ref, "--", path)
    changed = []
    for line in diff.splitlines():
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        payload = line[1:].strip()
        if not payload or STAMP_LIKE.search(payload):
            continue
        changed.append(line)
    return changed


def partition_conflicts(paths: list[str], base: str, ref: str) -> tuple[list, list]:
    """Separe les conflits verifiables des conflits exigeant un humain."""
    resolvable: list[str] = []
    blocking: list[tuple[str, list[str]]] = []
    for path in paths:
        if path in GENERATED:
            resolvable.append(path)
            continue
        lost = substantive_changes(base, ref, path)
        if lost:
            blocking.append((path, lost))
        else:
            resolvable.append(path)
    return resolvable, blocking


def find_conflict_markers() -> list[str]:
    """Fichiers suivis contenant encore un marqueur de conflit."""
    result = subprocess.run(
        [
            "git", "grep", "-l", "-I", "-E",
            r"^(<{7}|>{7})( |$)",
            "--", ".", ":(exclude)third_party",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode not in (0, 1):
        return []
    return [line for line in result.stdout.splitlines() if line]


def stamp_machine_sources(version: str) -> None:
    """Aligne les trois sources que possede semantic-release uniquement.

    [tool.semantic_release] ne declare que version_toml et version_variables :
    python/pyproject.toml, CMakeLists.txt et le LIBEMBEDDING_VERSION_STRING de
    config.h. scripts/sync_version.py ne les touche pas (il ne connait que les
    macros MAJOR/MINOR/PATCH et les ~70 en-tetes). Sans cette etape,
    python/pyproject.toml resterait sur la version du developpement,
    sync_version.py prendrait ce canon-la par defaut, et le controle final
    echouerait -- ou pire, tamponnerait tout le projet avec la mauvaise version.

    Les motifs sont ancres sur la ligne entiere et non sur un fragment : une
    forme inattendue ne trouve rien et leve une erreur, au lieu de laisser un
    fragment de l'ancienne ligne derriere le remplacement. C'est ce qui protege
    CMakeLists.txt d'un `set(...))` inexecutable.
    """
    targets = (
        (
            CANONICAL_TOML,
            re.compile(r'^version = "[^"]+"$', re.MULTILINE),
            f'version = "{version}"',
        ),
        (
            "CMakeLists.txt",
            re.compile(r'^set\(LIBEMBEDDING_VERSION "[^"]+"\)$', re.MULTILINE),
            f'set(LIBEMBEDDING_VERSION "{version}")',
        ),
        (
            "include/libembedding/config.h",
            re.compile(r'^#define LIBEMBEDDING_VERSION_STRING "[^"]+"$', re.MULTILINE),
            f'#define LIBEMBEDDING_VERSION_STRING "{version}"',
        ),
    )
    for relative, pattern, replacement in targets:
        path = REPO_ROOT / relative
        with open(path, encoding="utf-8", newline="") as handle:
            text = handle.read()
        if not pattern.search(text):
            raise SyncError(
                f"motif de version introuvable dans {relative} : "
                f"{pattern.pattern} (le format a peut-etre change)"
            )
        updated = pattern.sub(lambda _match: replacement, text, count=1)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
        print(f"  {relative} -> {version}")


def run_script(script: str, *args: str) -> None:
    """Lance un script du projet (outillage existant, pas de reimplementation)."""
    result = subprocess.run(
        [sys.executable, f"scripts/{script}", *args], cwd=REPO_ROOT
    )
    if result.returncode != 0:
        raise SyncError(f"scripts/{script} {' '.join(args)} : echec")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fusionne une branche et resout les conflits de tampon de version."
    )
    parser.add_argument(
        "--ref",
        default="origin/main",
        help="ref a fusionner dans la branche courante (defaut: origin/main)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="rapporte ce qui serait fait sans commiter (la fusion reste ouverte)",
    )
    parser.add_argument("--commit-message", default=None, help="message du commit")
    args = parser.parse_args()

    version = canonical_version(args.ref)
    base = git("merge-base", "HEAD", args.ref).strip()
    print(f"Version publiee sur {args.ref} : {version}")
    print(f"Base commune : {base[:8]}")

    if base == git("rev-parse", args.ref).strip():
        print(f"{args.ref} est deja integre. Rien a faire.")
        return 0

    head_before = git("rev-parse", "HEAD").strip()

    # --no-commit : on veut inspecter et verifier avant d'ecrire l'historique.
    merge = subprocess.run(
        ["git", "merge", "--no-commit", "--no-ff", args.ref],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if merge.returncode != 0 and not unmerged_paths():
        raise SyncError(f"git merge {args.ref} a echoue :\n{merge.stderr.strip()}")

    paths = unmerged_paths()
    resolvable, blocking = partition_conflicts(paths, base, args.ref)
    print(f"\n{len(paths)} fichier(s) en conflit")
    print(f"  {len(resolvable)} resolus automatiquement")
    print(f"  {len(blocking)} exigeant une decision humaine")

    if blocking:
        print("\nAbandon : la branche entrante a change du contenu.", file=sys.stderr)
        for path, lost in blocking:
            print(f"\n  {path}", file=sys.stderr)
            for line in lost[:4]:
                print(f"    {line}", file=sys.stderr)
            if len(lost) > 4:
                print(f"    ... et {len(lost) - 4} autre(s)", file=sys.stderr)
        print(
            "\nResoudre ces fichiers a la main (git checkout --ours/--theirs puis "
            "edition), commiter, puis relancer ce script : il reprendra au reste. "
            "Resoudre automatiquement ecraserait le contenu de main.",
            file=sys.stderr,
        )
        subprocess.run(["git", "merge", "--abort"], cwd=REPO_ROOT, capture_output=True)
        return 1

    if paths:
        print("\nResolution : contenu du developpement conserve, version reprise.")
        ours = [p for p in resolvable if p not in GENERATED]
        if ours:
            git("checkout", "--ours", "--", *ours)
            git("add", "--", *ours)
        for path in GENERATED:
            if path in resolvable:
                git("checkout", "--theirs", "--", path)

    if args.dry_run:
        print("\n(dry-run : aucun commit ; 'git merge --abort' pour annuler)")
        return 0

    # Les trois sources machines d'abord (semantic-release en est le seul
    # proprietaire), puis scripts/sync_version.py propage le canon vers les
    # ~70 en-tetes, les macros de config.h, _cdefs.h et AGENTS.md.
    print(f"\nAlignement des sources de verite sur {version}...")
    stamp_machine_sources(version)
    run_script("sync_version.py")
    git("add", "-A", "--", ".")

    if (REPO_ROOT / "scripts" / "generate_llm_docs.py").exists():
        print("Regeneration de llm.txt / llm_full.txt...")
        run_script("generate_llm_docs.py")
        git("add", "-A", "--", "llm.txt", "llm_full.txt")

    markers = find_conflict_markers()
    if markers:
        raise SyncError("marqueurs de conflit residuels : " + ", ".join(markers))

    run_script("sync_version.py", "--check")
    print("Tampons verifies : tous les X.Y.Z concordent.")

    if not git("diff", "--cached", "--name-only", "HEAD").splitlines():
        print("Aucun changement de contenu : fusion vide.")
        subprocess.run(["git", "merge", "--abort"], cwd=REPO_ROOT, capture_output=True)
        return 0

    message = args.commit_message or (
        f"chore(sync): integration de {args.ref} et alignement sur {version}\n\n"
        f"Conflits de tampon de version resolus automatiquement : le contenu vient\n"
        f"du developpement, la version vient de la release. La branche entrante\n"
        f"n'avait change que des numeros de version dans les fichiers traites.\n"
        f"Verifie par scripts/sync_branch.py."
    )
    git("commit", "--no-verify", "-m", message)
    print(
        f"\nFusion enregistree : {head_before[:8]} + {args.ref} -> {version}\n"
        f"Relisez le diff avant de pousser : git show --stat HEAD"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SyncError as exc:
        print(f"\nERREUR : {exc}", file=sys.stderr)
        sys.exit(1)