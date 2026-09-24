#!/usr/bin/env bash
#
# Single verification gate. Every phase must leave this green.
#
#   ./scripts/verify.sh              # everything
#   ./scripts/verify.sh --fast       # skip type checking and coverage
#   ./scripts/verify.sh --fix        # apply formatting and lint fixes first
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# A system-wide ROS install puts a Python 3.10 site-packages on PYTHONPATH,
# whose pytest plugins are imported ahead of ours and fail on 3.12.
export PYTHONPATH=""
export PYTHONDONTWRITEBYTECODE=1
export APP_ENV=test
export PA_ENV_FILE="${REPO_ROOT}/tests/.env.absent"

VENV="${REPO_ROOT}/.venv"
PY="${VENV}/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "No virtualenv at ${VENV}. Run: uv venv --python 3.12 .venv && uv pip install -e '.[dev]'"
    exit 1
fi

FAST=0
FIX=0
for arg in "$@"; do
    case "$arg" in
        --fast) FAST=1 ;;
        --fix) FIX=1 ;;
        *) echo "Unknown option: $arg"; exit 2 ;;
    esac
done

BOLD=$'\033[1m'; GREEN=$'\033[32m'; RED=$'\033[31m'; DIM=$'\033[2m'; RESET=$'\033[0m'
FAILURES=()
STEP=0

run_step() {
    local name="$1"; shift
    STEP=$((STEP + 1))
    printf '%s[%d] %s%s\n' "$BOLD" "$STEP" "$name" "$RESET"
    local output status
    output=$("$@" 2>&1); status=$?
    # pytest exits 5 when a suite has no tests yet, which is expected for a
    # directory a later phase will fill.
    if [[ $status -eq 5 && "$name" == *tests* ]]; then
        printf '    %s- no tests in this suite yet%s\n\n' "$DIM" "$RESET"
        return 0
    fi
    if [[ $status -eq 0 ]]; then
        printf '    %s✓ passed%s\n' "$GREEN" "$RESET"
        [[ -n "$output" ]] && printf '%s%s%s\n' "$DIM" "$(echo "$output" | tail -3 | sed 's/^/    /')" "$RESET"
    else
        printf '    %s✗ FAILED%s\n' "$RED" "$RESET"
        echo "$output" | tail -40 | sed 's/^/    /'
        FAILURES+=("$name")
    fi
    echo
}

if [[ $FIX -eq 1 ]]; then
    echo "${BOLD}Applying formatting and safe lint fixes${RESET}"
    "$VENV/bin/ruff" format .
    "$VENV/bin/ruff" check --fix .
    echo
fi

run_step "Format check (ruff format)"  "$VENV/bin/ruff" format --check .
run_step "Lint (ruff check)"           "$VENV/bin/ruff" check .

if [[ $FAST -eq 0 ]]; then
    run_step "Type check (mypy)" "$VENV/bin/mypy" app approvals database documents integrations \
        notifications proactive reachy_client scheduler security shared vision wardrobe \
        workers workstation cli
fi

run_step "Migrations (upgrade/downgrade round trip)" "$PY" scripts/check_migrations.py

run_step "Unit tests"        "$PY" -m pytest tests/unit -q
run_step "Integration tests" "$PY" -m pytest tests/integration -q
run_step "Contract tests"    "$PY" -m pytest tests/contract -q
run_step "Security tests"    "$PY" -m pytest tests/security -q
run_step "Acceptance tests"  "$PY" -m pytest tests/acceptance -q

if [[ $FAST -eq 0 ]]; then
    STEP=$((STEP + 1))
    printf '%s[%d] Coverage%s\n' "$BOLD" "$STEP" "$RESET"
    if coverage_output=$("$PY" -m pytest tests -q --cov --cov-report=term-missing:skip-covered 2>&1); then
        echo "$coverage_output" | grep -E '^(TOTAL|Required)' | sed 's/^/    /'
        printf '    %s✓ passed%s\n' "$GREEN" "$RESET"
    else
        printf '    %s✗ FAILED%s\n' "$RED" "$RESET"
        echo "$coverage_output" | tail -30 | sed 's/^/    /'
        FAILURES+=("Coverage")
    fi
    echo
fi

if [[ ${#FAILURES[@]} -eq 0 ]]; then
    printf '%s%sAll verification steps passed.%s\n' "$BOLD" "$GREEN" "$RESET"
    exit 0
fi

printf '%s%s%d step(s) failed:%s\n' "$BOLD" "$RED" "${#FAILURES[@]}" "$RESET"
printf '  - %s\n' "${FAILURES[@]}"
exit 1
