#!/usr/bin/env bash
#
# Cascadia Sentinel — Run All Demos
# ==================================
#
# One-command setup + run for all three instrument demos.
# Creates TWO venvs (main + moirai) to avoid gluonts version conflicts.
#
# Usage:
#   chmod +x run_all_demos.sh
#   ./run_all_demos.sh --self-test      # Synthetic data
#   ./run_all_demos.sh --real-data      # Pull live data
#   ./run_all_demos.sh --clean-install  # Wipe both venvs and rebuild

set -euo pipefail

VENV_DIR="${VENV_DIR:-./venv}"
MOIRAI_VENV="${MOIRAI_VENV:-./venv_moirai}"
PYTHON_CMD="${PYTHON_CMD:-python3}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

banner() {
    local text="$1"
    local width=70
    printf "\n%s\n" "$(printf '=%.0s' $(seq 1 $width))"
    printf "  %s\n" "$text"
    printf "%s\n\n" "$(printf '=%.0s' $(seq 1 $width))"
}

info()  { printf "${CYAN}[INFO]${NC} %s\n" "$1"; }
ok()    { printf "${GREEN}[OK]${NC}   %s\n" "$1"; }
warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$1"; }
error() { printf "${RED}[ERR]${NC}  %s\n" "$1" >&2; }
die()   { error "$1"; exit 1; }

check_python() {
    if ! command -v "$PYTHON_CMD" &> /dev/null; then
        die "Python command '$PYTHON_CMD' not found. Install Python 3.10+ and try again."
    fi
    local py_version
    py_version=$("$PYTHON_CMD" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    ok "Python $py_version detected"
}

create_venv() {
    local venv_path="$1"
    if [[ "${CLEAN_INSTALL:-0}" == "1" && -d "$venv_path" ]]; then
        info "Removing old venv at $venv_path ..."
        rm -rf "$venv_path"
    fi
    if [[ -d "$venv_path" ]]; then
        info "Virtual environment already exists at $venv_path"
    else
        info "Creating virtual environment in $venv_path ..."
        "$PYTHON_CMD" -m venv "$venv_path"
        ok "Created $venv_path"
    fi
}

activate_venv() {
    # shellcheck source=/dev/null
    source "$1/bin/activate"
}

upgrade_pip() {
    pip install --quiet --upgrade pip setuptools wheel
}

# ---------------------------------------------------------------------------
# Main venv deps (NO uni2ts here — that goes in venv_moirai)
# ---------------------------------------------------------------------------
install_main_deps() {
    banner "Installing Main Venv Dependencies"
    activate_venv "$VENV_DIR"
    upgrade_pip

    pip install --quiet numpy pandas scipy requests scikit-learn
    ok "Core packages installed"

    # PyTorch CPU
    if pip show torch &>/dev/null; then
        pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    fi
    pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
    ok "PyTorch (CPU) installed"

    # Seismic
    pip install --quiet obspy seisbench || warn "obspy/seisbench failed (optional)"

    # Weather: toto, chronos, timesfm — NO uni2ts
    pip install --quiet chronos-forecasting || warn "chronos-forecasting failed (optional)"

    if ! python -c "import timesfm" 2>/dev/null; then
        local tmpdir
        tmpdir=$(mktemp -d)
        if git clone --depth 1 https://github.com/google-research/timesfm.git "$tmpdir/timesfm" 2>/dev/null; then
            (cd "$tmpdir/timesfm" && pip install --quiet -e ".[torch]") || warn "TimesFM install failed"
        fi
        rm -rf "$tmpdir"
    fi

    local py_minor
    py_minor=$(python -c 'import sys; print(sys.version_info.minor)')
    if [[ "$py_minor" -ge 12 ]]; then
        pip install --quiet toto-models || warn "toto-models failed (optional)"
    fi

    pip install --quiet tirex-ts || warn "tirex-ts failed (optional)"
    ok "Main venv ready"
}

# ---------------------------------------------------------------------------
# Moirai venv deps (isolated gluonts 0.14.x + torch 2.4)
# ---------------------------------------------------------------------------
install_moirai_deps() {
    banner "Installing Moirai Venv Dependencies"
    create_venv "$MOIRAI_VENV"
    activate_venv "$MOIRAI_VENV"
    upgrade_pip

    pip install --quiet "uni2ts<3" "gluonts~=0.14.4" "torch~=2.4.0" numpy pandas scipy requests scikit-learn
    ok "Moirai venv ready at $MOIRAI_VENV"
}

# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------
run_magnetometer() {
    banner "MAGNETOMETER DEMO"
    local exit_code=0
    if [[ "$MODE" == "self-test" ]]; then
        python -m magnetometer.demos.magnetometer_demo --self-test || exit_code=$?
    else
        # Use May 2024 (known good data + calibrated storm period)
        # instead of Jan 2024 (70% gaps, triggers quality gate)
        python -m magnetometer.demos.magnetometer_demo \
            --fetch-real-data \
            --observatory VIC \
            --days 5 \
            --start-date 2024-05-08 \
            --warmup-days 3 \
            --cross-check-indices \
            --output-json magnetometer/results/magnetometer_results.json \
            --log-format json \
            || exit_code=$?
    fi

    if [[ $exit_code -eq 2 ]]; then
        warn "Magnetometer data-quality gate tripped — skipping metrics for this window."
        return 2
    elif [[ $exit_code -ne 0 ]]; then
        error "Magnetometer crashed (exit $exit_code)."
        return 1
    fi
    return 0
}

run_weather() {
    banner "WEATHER TSFM DEMO"

    local models="toto-22m,chronos2-small,timesfm25,moirai20-small,persistence,seasonal_naive"
    [[ "$MODE" == "self-test" ]] && models="persistence,seasonal_naive"

    # Split models
    local main_models="" moirai_models=""
    IFS=',' read -ra MODEL_ARRAY <<< "$models"
    for m in "${MODEL_ARRAY[@]}"; do
        if [[ "$m" == moirai20-* ]]; then
            moirai_models="${moirai_models}${m},"
        else
            main_models="${main_models}${m},"
        fi
    done
    main_models="${main_models%,}"
    moirai_models="${moirai_models%,}"

    local main_json="/tmp/weather_main.json"
    local moirai_json="/tmp/weather_moirai.json"

    # 1. Main env models
    if [[ -n "$main_models" ]]; then
        info "Main env models: $main_models"
        activate_venv "$VENV_DIR"
        python weather_tsfm_engine_v2_production_hybrid_fixed.py \
            --mode benchmark --model "$main_models" \
            --station-id 51337 --year 2024 --months 1 \
            --horizon 24 --n-splits 5 --output "$main_json"
    else
        echo '{"benchmark_metrics":{}}' > "$main_json"
    fi

    # 2. Moirai venv models
    if [[ -n "$moirai_models" ]]; then
        if [[ ! -d "$MOIRAI_VENV" ]]; then
            warn "Moirai venv missing — skipping Moirai models"
            echo '{"benchmark_metrics":{}}' > "$moirai_json"
        else
            info "Moirai venv models: $moirai_models"
            "$MOIRAI_VENV/bin/python" weather_tsfm_engine_v2_production_hybrid_fixed.py \
                --mode benchmark --model "$moirai_models" \
                --station-id 51337 --year 2024 --months 1 \
                --horizon 24 --n-splits 5 --output "$moirai_json"
        fi
    else
        echo '{"benchmark_metrics":{}}' > "$moirai_json"
    fi

    # 3. Merge and print
    info "Merging results ..."
    "$PYTHON_CMD" -c "
import json
with open('$main_json') as f: main = json.load(f)
with open('$moirai_json') as f: moirai = json.load(f)
for field, backends in moirai.get('benchmark_metrics', {}).items():
    if field not in main.get('benchmark_metrics', {}):
        main['benchmark_metrics'][field] = {}
    main['benchmark_metrics'][field].update(backends)
with open('weather_merged_results.json', 'w') as f:
    json.dump(main, f, indent=2)

print('\n' + '='*120)
print(f\"{'field':<14}{'backend':<22}{'params':<8}{'MAE':<10}{'RMSE':<10}{'MASE':<10}{'CRPS':<10}{'ms/call':<10}{'mv':<4}\")
print('-'*120)
for feat, backends in main.get('benchmark_metrics', {}).items():
    ranked = sorted(backends.items(), key=lambda kv: kv[1].get('avg_mase', 999))
    for bname, m in ranked:
        mv = 'yes' if m.get('multivariate') else 'no'
        print(f\"{feat:<14}{bname:<22}{str(m.get('params','-')):<8}\"
              f\"{m.get('avg_mae','-'):<10}{m.get('avg_rmse','-'):<10}\"
              f\"{m.get('avg_mase','-'):<10}{m.get('avg_crps','-'):<10}\"
              f\"{m.get('avg_latency_ms','-'):<10}{mv:<4}\")
print('='*120)
print('MASE < 1.0 beats seasonal-naive. CRPS = probabilistic calibration (lower = better).')
print('mv = multivariate native. -mv suffix = Toto multivariate on clean sensors.')
print(\"\n⚠️  NOTE: ECCC data may overlap Chronos/TimesFM pretraining — treat those MASE as optimistic.\n\")
"
}

run_seismic() {
    banner "SEISMIC DEMO"
    export CUDA_VISIBLE_DEVICES=""
    if [[ "$MODE" == "self-test" ]]; then
        python seismic_demo.py --self-test
    else
        python seismic_demo.py --fetch-real-data \
            --network IU --station MAJO --channel BH? \
            --start 2024-01-01T07:00:00 --end 2024-01-01T11:00:00 \
            --window-s 60 --step-s 30 --prob-threshold 0.4
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    MODE=""
    INSTALL_ONLY=0
    CLEAN_INSTALL=0

    for arg in "$@"; do
        case "$arg" in
            --self-test) MODE="self-test" ;;
            --real-data) MODE="real-data" ;;
            --install-only) INSTALL_ONLY=1 ;;
            --clean-install) CLEAN_INSTALL=1 ;;
            --help|-h)
                cat <<'EOF'
Usage:
  ./run_all_demos.sh --self-test       Synthetic data
  ./run_all_demos.sh --real-data       Real data from public archives
  ./run_all_demos.sh --install-only    Install deps only
  ./run_all_demos.sh --clean-install   Wipe venvs and rebuild
EOF
                exit 0
                ;;
        esac
    done

    if [[ -z "$MODE" && "$INSTALL_ONLY" -eq 0 ]]; then
        die "No mode specified. Use --self-test or --real-data. See --help."
    fi

    banner "Cascadia Sentinel — Pre-flight Checks"
    check_python

    create_venv "$VENV_DIR"
    create_venv "$MOIRAI_VENV"

    install_main_deps
    install_moirai_deps

    if [[ "$INSTALL_ONLY" -eq 1 ]]; then
        banner "Install Complete"
        ok "Run with --self-test or --real-data when ready."
        exit 0
    fi

    banner "Cascadia Sentinel — All Demos ($MODE)"

    local e1=0 e2=0 e3=0
    run_magnetometer || e1=$?
    run_weather      || e2=$?
    run_seismic      || e3=$?

    banner "RUN SUMMARY"
    printf "Mode:        %s\n" "$MODE"
    printf "Main venv:   %s\n" "$VENV_DIR"
    printf "Moirai venv: %s\n" "$MOIRAI_VENV"

    local any_error=0
    if [[ $e1 -eq 2 ]]; then
        warn "Magnetometer: data-quality gate tripped (degraded, not fatal)"
    elif [[ $e1 -ne 0 ]]; then
        error "Magnetometer: failed (exit $e1)"
        any_error=1
    else
        ok "Magnetometer: passed"
    fi

    if [[ $e2 -ne 0 ]]; then
        error "Weather: failed (exit $e2)"
        any_error=1
    else
        ok "Weather: passed"
    fi

    if [[ $e3 -ne 0 ]]; then
        error "Seismic: failed (exit $e3)"
        any_error=1
    else
        ok "Seismic: passed"
    fi

    if [[ $any_error -ne 0 ]]; then
        warn "One or more demos exited with an error."
        exit 1
    fi
    ok "All demos completed successfully."
}

main "$@"
