#!/usr/bin/env bash
set -euo pipefail

# 绝对路径（根据当前仓库位置固定）
BASE_DIR="/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG"
ENTRY_PY="$BASE_DIR/UltrasoundRAG/UltrasoundRAG/ultrasoundrag.py"

# 可选参数
DB_NAME="default"
TOP_K=5

echo "[1/3] 重建所有索引（Markdown + Image，删除重建）"
python "$ENTRY_PY" --build-md --build-image --recreate

echo "[2/3] 运行四种检索模式冒烟测试（重点关注 T2T / T2I）"
python "$ENTRY_PY" --test-modes --db-name "$DB_NAME" --top-k $TOP_K

echo "[3/3] 冒烟测试完成"
echo "完成。"


