#!/bin/zsh
set -u
cd -- "${0:A:h}" || exit 1
export PYTHONDONTWRITEBYTECODE=1
trap 'unset OPENAI_API_KEY' EXIT

pause_and_exit() {
    local result_code="$1"
    read -r '?Press Return to close this window. '
    exit "$result_code"
}

if ! command -v python3 >/dev/null 2>&1; then
    print "Python 3 is required."
    pause_and_exit 1
fi

if [[ -d astra-api-trial-02 ]]; then
    print "An attempt is already recorded in astra-api-trial-02."
    pause_and_exit 1
fi

if ! python3 - <<'PY'
import hashlib
from pathlib import Path
path = Path('prompt-preview.txt')
expected = 'a2c83e67e0ad03b63c902690652f37d517189d43b5562339f46d359a91455b26'
if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
    raise SystemExit('The prepared prompt has changed. Check it before running the trial.')
PY
then
    pause_and_exit 1
fi

if ! python3 run_astra_api.py; then
    pause_and_exit 1
fi

if [[ "${1:-}" == "--check" ]]; then
    exit 0
fi

print ""
print 'GPT-6 Astra. Second recorded attempt. Input is billed again.'
print "Reasoning effort high. Maximum output 32768 tokens."
read -rs 'OPENAI_API_KEY?Paste your OpenAI API key and press Return to start: '
print ""
if [[ -z "$OPENAI_API_KEY" ]]; then
    print "No key entered."
    pause_and_exit 1
fi
export OPENAI_API_KEY
print "Request starting. Keep this window open while Astra responds."
python3 -u run_astra_api.py --live
result_code=$?
unset OPENAI_API_KEY
if [[ "$result_code" -eq 0 ]]; then
    print ""
    cat astra-api-trial-02/answer.json
else
    print ""
    print "The request did not finish successfully. Share the error message before taking another step."
fi
pause_and_exit "$result_code"
