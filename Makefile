# UltrasoundRAG 项目 Makefile
# 提供常用的开发、测试和部署命令

.PHONY: help install install-dev clean test lint format build docs serve-docs

# 默认目标
help:
	@echo "UltrasoundRAG 开发工具命令："
	@echo ""
	@echo "安装和环境:"
	@echo "  install      - 安装包（生产模式）"
	@echo "  install-dev  - 安装包（开发模式，包含开发依赖）"
	@echo "  clean        - 清理构建文件和缓存"
	@echo ""
	@echo "代码质量:"
	@echo "  test         - 运行所有测试"
	@echo "  test-unit    - 运行单元测试"
	@echo "  test-integration - 运行集成测试"
	@echo "  lint         - 运行代码检查（flake8, mypy）"
	@echo "  format       - 格式化代码（black, isort）"
	@echo "  check        - 检查代码格式（不修改）"
	@echo ""
	@echo "构建和发布:"
	@echo "  build        - 构建分发包"
	@echo "  docs         - 构建文档"
	@echo "  serve-docs   - 本地预览文档"
	@echo ""
	@echo "UltrasoundRAG 功能:"
	@echo "  demo         - 运行快速演示"
	@echo "  index        - 构建所有索引"
	@echo "  benchmark    - 运行基准测试"

# === 安装和环境 ===

install:
	pip install .

install-dev:
	pip install -e ".[dev,test,docs]"

clean:
	@echo "清理构建文件和缓存..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	@echo "清理完成"

# === 代码质量 ===

test:
	@echo "运行所有测试..."
	pytest

test-unit:
	@echo "运行单元测试..."
	pytest tests/unit/ -v

test-integration:
	@echo "运行集成测试..."
	pytest tests/integration/ -v

test-e2e:
	@echo "运行端到端测试..."
	pytest tests/e2e/ -v

lint:
	@echo "运行代码检查..."
	flake8 ultrasoundrag/ tests/
	mypy ultrasoundrag/

format:
	@echo "格式化代码..."
	black ultrasoundrag/ tests/
	isort ultrasoundrag/ tests/

check:
	@echo "检查代码格式..."
	black --check ultrasoundrag/ tests/
	isort --check-only ultrasoundrag/ tests/

# === 构建和文档 ===

build: clean
	@echo "构建分发包..."
	python -m build

docs:
	@echo "构建文档..."
	cd docs && make html

serve-docs:
	@echo "启动文档服务器..."
	cd docs/_build/html && python -m http.server 8000

# === UltrasoundRAG 功能 ===

demo:
	@echo "运行UltrasoundRAG快速演示..."
	python -c "from ultrasoundrag import quick_demo; quick_demo()"

index:
	@echo "构建所有索引..."
	python -m ultrasoundrag index --build-all

index-md:
	@echo "构建Markdown索引..."
	python -m ultrasoundrag index --build-md

index-img:
	@echo "构建图像索引..."
	python -m ultrasoundrag index --build-img

search:
	@echo "测试搜索功能..."
	python -m ultrasoundrag search "心脏超声检查"

test-modes:
	@echo "测试所有检索模式..."
	python -m ultrasoundrag test --mode all

benchmark:
	@echo "运行基准测试..."
	python -m ultrasoundrag benchmark --queries "心脏超声,肝脏检查,胎儿发育"

# === 开发环境设置 ===

setup-dev: install-dev
	@echo "设置开发环境..."
	pre-commit install
	@echo "开发环境设置完成"

# === CI/CD 相关 ===

ci-test: lint test
	@echo "CI测试流程完成"

ci-build: clean build
	@echo "CI构建流程完成"

# === 版本管理 ===

version:
	@echo "当前版本信息:"
	@python -c "from ultrasoundrag import __version__; print(f'UltrasoundRAG v{__version__}')"

# === 帮助信息 ===

show-tree:
	@echo "项目结构:"
	tree ultrasoundrag/ -I "__pycache__|*.pyc|*.pyo"
