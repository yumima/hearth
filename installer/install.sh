#!/bin/sh
# Install hearth, a local AI engine, and mantel, its chat window — for this user,
# without administrator rights.
#
#   curl -fsSL https://yumima.github.io/hearth/install.sh | sh
#   … | sh -s -- --no-mantel --no-autostart
#
# Everything goes into one folder, and `hearth uninstall` removes it. An Ollama,
# a hearth or a Python that is already here is not touched. This script fetches
# uv and hearth, checks both against the hashes below, and lets hearth do the rest.
#
# In functions, run at the end: a download that is cut off runs nothing.
set -eu

VERSION="@VERSION@"
BASE_URL="@BASE_URL@"
MANIFEST_URL="@MANIFEST_URL@"
MANIFEST_SHA256="@MANIFEST_SHA256@"
PYTHON="@PYTHON@"
WHEEL="@WHEEL@"
WHEEL_SHA256="@WHEEL_SHA256@"
REQUIREMENTS_SHA256="@REQUIREMENTS_SHA256@"
UV_VERSION="@UV_VERSION@"

say() { printf '%s\n' "$*"; }
fail() { printf 'hearth: %s\n' "$*" >&2; exit 1; }

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# fetch URL FILE SHA256
fetch() {
    if [ -f "$2" ] && [ "$(sha256 "$2")" = "$3" ]; then return 0; fi
    curl -fsSL --proto '=https,http' --retry 3 --connect-timeout 20 -o "$2.part" "$1" || fail "could not download $1"
    [ "$(sha256 "$2.part")" = "$3" ] || { rm -f "$2.part"; fail "$1 is not the published file"; }
    mv "$2.part" "$2"
}

main() {
    command -v curl >/dev/null 2>&1 || fail "curl is needed"
    command -v tar >/dev/null 2>&1 || fail "tar is needed"
    [ "$(id -u)" != 0 ] || fail "run this as yourself, not as root: it installs for one user"

    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)
            UV_FILE=uv-x86_64-unknown-linux-gnu.tar.gz UV_SHA256="@UV_LINUX_X86_64@"
            case "${XDG_DATA_HOME:-}" in /*) ROOT="$XDG_DATA_HOME/hearth" ;; *) ROOT="$HOME/.local/share/hearth" ;; esac ;;
        Darwin-arm64)
            UV_FILE=uv-aarch64-apple-darwin.tar.gz UV_SHA256="@UV_MACOS_ARM64@"
            ROOT="$HOME/Library/Application Support/hearth" ;;
        *) fail "there is no hearth for $(uname -s) on $(uname -m) yet" ;;
    esac
    ROOT="${HEARTH_DATA:-$ROOT}"
    if [ -e "$ROOT" ] && [ ! -e "$ROOT/.hearth-install" ] && [ -n "$(ls -A "$ROOT" 2>/dev/null)" ] && [ ! -d "$ROOT/versions" ]; then
        # models and voices an uninstall was asked to keep are fine; anything else is not ours
        for f in "$ROOT"/* ; do
            case "${f##*/}" in models|voices) ;; *) fail "$ROOT is there already and was not made by this installer" ;; esac
        done
    fi

    # (hearth looks for more of these itself; this one saves the download.)
    for name in hearth mantel; do
        link="$HOME/.local/bin/$name"
        if [ -h "$link" ]; then
            case "$(readlink "$link")" in
                "$ROOT"/*) ;;
                *) fail "$link is another $name ($(readlink "$link")); hearth was not installed, so as not to disturb it" ;;
            esac
        fi
    done

    say "Installing hearth $VERSION in $ROOT"
    DOWN="$ROOT/downloads/$VERSION"
    mkdir -p "$ROOT/bin" "$DOWN"
    printf "hearth's installer made this folder, and removing hearth removes it.\n" > "$ROOT/.hearth-install"

    fetch "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$UV_FILE" "$DOWN/$UV_FILE" "$UV_SHA256"
    tar -xzf "$DOWN/$UV_FILE" -C "$DOWN"
    mv -f "$DOWN/${UV_FILE%.tar.gz}/uv" "$ROOT/bin/uv"

    fetch "${BASE_URL}${WHEEL}" "$DOWN/$WHEEL" "$WHEEL_SHA256"
    fetch "${BASE_URL}hearth-requirements.txt" "$DOWN/hearth-requirements.txt" "$REQUIREMENTS_SHA256"

    # Nothing of the person's own Python or uv settings has a part in it.
    unset VIRTUAL_ENV PYTHONPATH PYTHONHOME PIP_INDEX_URL UV_INDEX_URL UV_EXTRA_INDEX_URL UV_INDEX || true
    export UV_PYTHON_INSTALL_DIR="$ROOT/python" UV_CACHE_DIR="$ROOT/cache" UV_NO_CONFIG=1 \
        UV_PYTHON_PREFERENCE=only-managed UV_LINK_MODE=copy UV_DEFAULT_INDEX=https://pypi.org/simple
    ENV="$ROOT/versions/$VERSION/hearth"
    if [ ! -f "$ENV/.complete" ]; then
        rm -rf "$ENV"
        say "Installing Python…"
        "$ROOT/bin/uv" python install --quiet "$PYTHON" || fail "Python could not be installed"
        "$ROOT/bin/uv" venv --quiet --python "$PYTHON" "$ENV" || fail "the environment could not be made"
        say "Installing hearth…"
        "$ROOT/bin/uv" pip install --quiet --python "$ENV" --require-hashes --only-binary :all: -r "$DOWN/hearth-requirements.txt" \
            || fail "what hearth needs could not be installed"
        "$ROOT/bin/uv" pip install --quiet --python "$ENV" --no-deps --no-index "$DOWN/$WHEEL" || fail "hearth could not be installed"
        : > "$ENV/.complete"
    fi
    status=0
    "$ENV/bin/python" -I -m hearth.install finish --release "$MANIFEST_URL" --sha256 "$MANIFEST_SHA256" "$@" || status=$?
    if [ "$status" = 3 ] && [ ! -e "$ROOT/current.json" ]; then
        # It stepped aside for something that is here: so does what was brought for it.
        for part in bin versions python cache downloads .hearth-install; do rm -rf "${ROOT:?}/$part"; done
        rmdir "$ROOT" 2>/dev/null || true
    fi
    return "$status"
}

main "$@"
