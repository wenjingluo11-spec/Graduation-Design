#!/usr/bin/env bash
set -euo pipefail
PY=${PYTHON:-python3}
echo "==> 创建 .venv-mac 虚拟环境"
$PY -m venv .venv-mac
# shellcheck disable=SC1091
source .venv-mac/bin/activate
echo "==> 升级 pip"
pip install -U pip
echo "==> 安装依赖"
pip install -r requirements.txt
echo "==> 下载 NLTK punkt"
python -c "import nltk; nltk.download('punkt', quiet=True)"
echo "✓ Mac 环境就绪。 source .venv-mac/bin/activate 即可使用。"
