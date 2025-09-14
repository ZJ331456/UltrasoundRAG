# """
# 简单的API集成测试：启动本地FastAPI服务并发起请求验证检索效果。

# 运行方式：
#   1) 启动服务（一个终端）：
#      uvicorn api:app --host 0.0.0.0 --port 8000
#   2) 运行本测试（另一个终端）：
#      python -m test.test_api
# """

# import json
# import time
# import requests


# BASE_URL = "http://127.0.0.1:8000"


# def pretty(obj):
#     return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


# def wait_server(timeout_sec: int = 10) -> None:
#     start = time.time()
#     while time.time() - start < timeout_sec:
#         try:
#             r = requests.get(BASE_URL + "/databases", timeout=1)
#             if r.status_code == 200:
#                 return
#         except Exception:
#             time.sleep(0.5)
#     raise RuntimeError("API服务未就绪，请先启动: uvicorn UltrasoundRAG.api:app --port 8000")


# def test_list_databases():
#     r = requests.get(BASE_URL + "/databases", timeout=10)
#     r.raise_for_status()
#     data = r.json()
#     print("/databases =>", pretty(data))


# def test_t2t():
#     payload = {"db_name": "default", "mode": "t2t", "query": "心脏超声检查方法", "top_k": 3}
#     r = requests.post(BASE_URL + "/search", json=payload, timeout=60)
#     r.raise_for_status()
#     data = r.json()
#     print("/search t2t =>", pretty(data))


# def test_t2i():
#     payload = {"db_name": "default", "mode": "t2i", "query": "四腔心切面", "top_k": 3}
#     r = requests.post(BASE_URL + "/search", json=payload, timeout=60)
#     r.raise_for_status()
#     data = r.json()
#     print("/search t2i =>", pretty(data))


# def test_i2t():
#     payload = {"db_name": "default", "mode": "i2t", "image_path": "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/image/07_腹部超声/图片00008.jpg", "top_k": 3}
#     r = requests.post(BASE_URL + "/search", json=payload, timeout=60)
#     r.raise_for_status()
#     data = r.json()
#     print("/search i2t =>", pretty(data))


# def test_i2i():
#     payload = {"db_name": "default", "mode": "i2i", "image_path": "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/image/07_腹部超声/图片00008.jpg", "top_k": 3}
#     r = requests.post(BASE_URL + "/search", json=payload, timeout=60)
#     r.raise_for_status()
#     data = r.json()
#     print("/search i2i =>", pretty(data))


# def main():
#     wait_server()
#     test_list_databases()
#     test_t2t()
#     test_t2i()
#     test_i2t()
#     test_i2i()


# if __name__ == "__main__":
#     main()


import requests
import json
import os
from datetime import datetime

# 创建保存目录
SAVE_DIR = "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/result/test_api_response"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_response(response, test_name, request_data):
    """保存API响应为JSON文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{test_name}_{timestamp}.json"
    filepath = os.path.join(SAVE_DIR, filename)
    
    # 准备保存的数据
    save_data = {
        "test_name": test_name,
        "timestamp": timestamp,
        "request": request_data,
        "response": {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "data": response.json() if response.status_code == 200 else response.text
        }
    }
    
    # 保存为JSON文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"✅ {test_name} 响应已保存到: {filepath}")
    return filepath

# 文本搜索测试
print("🔍 执行文本搜索测试...")
t2t_request = {
    "mode": "t2t",
    "query": "眼球壁包括什么？",
    "top_k": 10,
    "db_name": "default"
}
t2t_response = requests.post("http://localhost:8001/search", json=t2t_request)
print("T2T响应:")
print(json.dumps(t2t_response.json(), ensure_ascii=False, indent=2))
save_response(t2t_response, "t2t_search", t2t_request)

print("=" * 50)

# 图像搜索测试
print("🖼️ 执行图像搜索测试...")
i2t_request = {
    "mode": "i2t", 
    "image_path": "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/image/02_实用浅表/图片00001.jpg",
    "top_k": 5,
    "db_name": "default"
}
i2t_response = requests.post("http://localhost:8001/search", json=i2t_request)
print("I2T响应:")
print(json.dumps(i2t_response.json(), ensure_ascii=False, indent=2))
save_response(i2t_response, "i2t_search", i2t_request)

print("=" * 50)
print(f"📁 所有响应结果已保存到: {SAVE_DIR}")