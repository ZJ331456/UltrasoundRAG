"""
简单的API集成测试：启动本地FastAPI服务并发起请求验证检索效果。

运行方式：
  1) 启动服务（一个终端）：
     uvicorn api:app --host 0.0.0.0 --port 8000
  2) 运行本测试（另一个终端）：
     python -m test.test_api
"""

import json
import time
import requests


BASE_URL = "http://127.0.0.1:8000"


def pretty(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def wait_server(timeout_sec: int = 10) -> None:
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            r = requests.get(BASE_URL + "/databases", timeout=1)
            if r.status_code == 200:
                return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("API服务未就绪，请先启动: uvicorn UltrasoundRAG.api:app --port 8000")


def test_list_databases():
    r = requests.get(BASE_URL + "/databases", timeout=10)
    r.raise_for_status()
    data = r.json()
    print("/databases =>", pretty(data))


def test_t2t():
    payload = {"db_name": "default", "mode": "t2t", "query": "心脏超声检查方法", "top_k": 3}
    r = requests.post(BASE_URL + "/search", json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    print("/search t2t =>", pretty(data))


def test_t2i():
    payload = {"db_name": "default", "mode": "t2i", "query": "四腔心切面", "top_k": 3}
    r = requests.post(BASE_URL + "/search", json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    print("/search t2i =>", pretty(data))


def test_i2t():
    payload = {"db_name": "default", "mode": "i2t", "image_path": "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/image/07_腹部超声/图片00008.jpg", "top_k": 3}
    r = requests.post(BASE_URL + "/search", json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    print("/search i2t =>", pretty(data))


def test_i2i():
    payload = {"db_name": "default", "mode": "i2i", "image_path": "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/image/07_腹部超声/图片00008.jpg", "top_k": 3}
    r = requests.post(BASE_URL + "/search", json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    print("/search i2i =>", pretty(data))


def main():
    wait_server()
    test_list_databases()
    test_t2t()
    test_t2i()
    test_i2t()
    test_i2i()


if __name__ == "__main__":
    main()


