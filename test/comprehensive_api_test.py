#!/usr/bin/env python3
"""
全面的API测试脚本
测试所有可用的API接口并保存响应结果

运行方式：
python -m test.comprehensive_api_test
"""

import requests
import json
import os
import time
from datetime import datetime
from typing import Dict, Any

# 配置
BASE_URL = "http://localhost:8001"
SAVE_DIR = "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/result/test_api_response"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_response(response, test_name: str, request_data: Dict[str, Any] = None):
    """保存API响应为JSON文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{test_name}_{timestamp}.json"
    filepath = os.path.join(SAVE_DIR, filename)
    
    # 准备保存的数据
    save_data = {
        "test_name": test_name,
        "timestamp": timestamp,
        "request": request_data or {},
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

def test_health_check():
    """测试健康检查接口"""
    print("🏥 测试健康检查接口...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        save_response(response, "health_check")
        return True
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False

def test_system_status():
    """测试系统状态接口"""
    print("📊 测试系统状态接口...")
    try:
        response = requests.get(f"{BASE_URL}/system/status", timeout=10)
        print(f"状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        save_response(response, "system_status")
        return True
    except Exception as e:
        print(f"❌ 系统状态获取失败: {e}")
        return False

def test_list_databases():
    """测试数据库列表接口"""
    print("🗄️ 测试数据库列表接口...")
    try:
        response = requests.get(f"{BASE_URL}/databases", timeout=10)
        print(f"状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        save_response(response, "list_databases")
        return True
    except Exception as e:
        print(f"❌ 数据库列表获取失败: {e}")
        return False

def test_t2t_search():
    """测试文本到文本搜索"""
    print("🔍 测试文本到文本搜索...")
    request_data = {
        "mode": "t2t",
        "query": "心脏超声检查方法",
        "top_k": 5,
        "db_name": "default"
    }
    try:
        response = requests.post(f"{BASE_URL}/search", json=request_data, timeout=60)
        print(f"状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        save_response(response, "t2t_search", request_data)
        return True
    except Exception as e:
        print(f"❌ T2T搜索失败: {e}")
        return False

def test_t2i_search():
    """测试文本到图像搜索"""
    print("🖼️ 测试文本到图像搜索...")
    request_data = {
        "mode": "t2i",
        "query": "四腔心切面",
        "top_k": 3,
        "db_name": "default"
    }
    try:
        response = requests.post(f"{BASE_URL}/search", json=request_data, timeout=60)
        print(f"状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        save_response(response, "t2i_search", request_data)
        return True
    except Exception as e:
        print(f"❌ T2I搜索失败: {e}")
        return False

def test_i2t_search():
    """测试图像到文本搜索"""
    print("🔍 测试图像到文本搜索...")
    request_data = {
        "mode": "i2t",
        "image_path": "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/image/02_实用浅表/图片00001.jpg",
        "top_k": 3,
        "db_name": "default"
    }
    try:
        response = requests.post(f"{BASE_URL}/search", json=request_data, timeout=60)
        print(f"状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        save_response(response, "i2t_search", request_data)
        return True
    except Exception as e:
        print(f"❌ I2T搜索失败: {e}")
        return False

def test_i2i_search():
    """测试图像到图像搜索"""
    print("🖼️ 测试图像到图像搜索...")
    request_data = {
        "mode": "i2i",
        "image_path": "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/image/02_实用浅表/图片00001.jpg",
        "top_k": 3,
        "db_name": "default"
    }
    try:
        response = requests.post(f"{BASE_URL}/search", json=request_data, timeout=60)
        print(f"状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        save_response(response, "i2i_search", request_data)
        return True
    except Exception as e:
        print(f"❌ I2I搜索失败: {e}")
        return False

def test_multimodal_search():
    """测试多模态搜索"""
    print("🔄 测试多模态搜索...")
    request_data = {
        "mode": "multimodal",
        "query": "心脏超声诊断",
        "top_k": 3,
        "db_name": "default"
    }
    try:
        response = requests.post(f"{BASE_URL}/search", json=request_data, timeout=60)
        print(f"状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        save_response(response, "multimodal_search", request_data)
        return True
    except Exception as e:
        print(f"❌ 多模态搜索失败: {e}")
        return False

def test_auto_search():
    """测试自动模式搜索"""
    print("🤖 测试自动模式搜索...")
    request_data = {
        "mode": "auto",
        "query": "肝脏病变诊断",
        "top_k": 3,
        "db_name": "default"
    }
    try:
        response = requests.post(f"{BASE_URL}/search", json=request_data, timeout=60)
        print(f"状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        save_response(response, "auto_search", request_data)
        return True
    except Exception as e:
        print(f"❌ 自动搜索失败: {e}")
        return False

def test_mixed_search():
    """测试图文混合搜索"""
    print("🔄 测试图文混合搜索...")
    request_data = {
        "mode": "mixed",
        "query": "甲状腺结节在哪？请分析这张图片中的异常区域",
        "image_path": "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/image/02_实用浅表/图片00001.jpg",
        "top_k": 3,
        "db_name": "default"
    }
    try:
        response = requests.post(f"{BASE_URL}/search", json=request_data, timeout=90)
        print(f"状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        save_response(response, "mixed_search", request_data)
        return True
    except Exception as e:
        print(f"❌ 图文混合搜索失败: {e}")
        return False

def test_admin_endpoints():
    """测试管理接口"""
    print("⚙️ 测试管理接口...")
    
    # 测试清空缓存
    try:
        response = requests.post(f"{BASE_URL}/admin/cache/clear", timeout=10)
        print(f"清空缓存状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        save_response(response, "admin_clear_cache")
    except Exception as e:
        print(f"❌ 清空缓存失败: {e}")
    
    # 测试重新加载模型
    try:
        response = requests.post(f"{BASE_URL}/admin/models/reload", timeout=10)
        print(f"重新加载模型状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        save_response(response, "admin_reload_models")
    except Exception as e:
        print(f"❌ 重新加载模型失败: {e}")
    
    # 测试获取告警
    try:
        response = requests.get(f"{BASE_URL}/admin/alerts", timeout=10)
        print(f"获取告警状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        save_response(response, "admin_alerts")
    except Exception as e:
        print(f"❌ 获取告警失败: {e}")

def main():
    """主测试函数"""
    print("🚀 开始全面API测试")
    print(f"📁 结果将保存到: {SAVE_DIR}")
    print("=" * 60)
    
    # 测试结果统计
    test_results = {}
    
    # 基础接口测试
    test_results["health_check"] = test_health_check()
    print("\n" + "=" * 60)
    
    test_results["system_status"] = test_system_status()
    print("\n" + "=" * 60)
    
    test_results["list_databases"] = test_list_databases()
    print("\n" + "=" * 60)
    
    # 搜索接口测试
    test_results["t2t_search"] = test_t2t_search()
    print("\n" + "=" * 60)
    
    test_results["t2i_search"] = test_t2i_search()
    print("\n" + "=" * 60)
    
    test_results["i2t_search"] = test_i2t_search()
    print("\n" + "=" * 60)
    
    test_results["i2i_search"] = test_i2i_search()
    print("\n" + "=" * 60)
    
    test_results["multimodal_search"] = test_multimodal_search()
    print("\n" + "=" * 60)
    
    test_results["auto_search"] = test_auto_search()
    print("\n" + "=" * 60)
    
    test_results["mixed_search"] = test_mixed_search()
    print("\n" + "=" * 60)
    
    # 管理接口测试
    test_admin_endpoints()
    print("\n" + "=" * 60)
    
    # 输出测试总结
    print("📊 测试总结:")
    print(f"✅ 成功: {sum(test_results.values())}")
    print(f"❌ 失败: {len(test_results) - sum(test_results.values())}")
    print(f"📁 所有响应结果已保存到: {SAVE_DIR}")
    
    # 保存测试总结
    summary_data = {
        "test_summary": {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(test_results),
            "successful_tests": sum(test_results.values()),
            "failed_tests": len(test_results) - sum(test_results.values()),
            "test_results": test_results
        }
    }
    
    summary_file = os.path.join(SAVE_DIR, f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    print(f"📋 测试总结已保存到: {summary_file}")

if __name__ == "__main__":
    main()
