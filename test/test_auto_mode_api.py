#!/usr/bin/env python3
"""
Auto模式API测试脚本
使用指定的文本和图片查询，测试auto模式的多模态检索功能
"""

import requests
import json
import time
import os
from datetime import datetime
from typing import Dict, Any

class AutoModeAPITester:
    """Auto模式API测试器"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        
        # 测试参数
        self.test_config = {
            "text_query": "这张图片是什么意思？",
            "image_path": "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/image/01_超声标准切面/图片00001.jpg",
            "mode": "auto",
            "text_collections": ["normal_book_md", "thyroid_agent_md"],
            "image_collections": ["normal_book_image"],
            "top_k": 10
        }
    
    def check_server_status(self) -> bool:
        """检查服务器状态"""
        try:
            print("正在检查服务器状态...")
            response = requests.get(f"{self.base_url}/api/v1/rag/health", timeout=10)
            print(f"服务器响应状态: {response.status_code}")
            if response.status_code == 200:
                print("✅ 服务器连接正常")
                return True
            else:
                print(f"❌ 服务器响应异常: {response.text}")
                return False
        except Exception as e:
            print(f"❌ 服务器连接失败: {e}")
            return False
    
    def check_image_file(self) -> bool:
        """检查图片文件是否存在"""
        image_path = self.test_config["image_path"]
        if os.path.exists(image_path):
            file_size = os.path.getsize(image_path)
            print(f"✅ 图片文件存在: {image_path}")
            print(f"   文件大小: {file_size} 字节")
            return True
        else:
            print(f"❌ 图片文件不存在: {image_path}")
            return False
    
    def test_auto_mode_api(self) -> Dict[str, Any]:
        """测试auto模式API - 使用curl命令"""
        print("\n" + "="*80)
        print("测试Auto模式API (使用curl命令)")
        print("="*80)
        print(f"文本查询: {self.test_config['text_query']}")
        print(f"图片路径: {self.test_config['image_path']}")
        print(f"模式: {self.test_config['mode']}")
        print(f"文本集合: {self.test_config['text_collections']}")
        print(f"图片集合: {self.test_config['image_collections']}")
        print(f"Top-K: {self.test_config['top_k']}")
        print("="*80)
        
        try:
            import subprocess
            import tempfile
            
            # 构建curl命令
            curl_cmd = [
                'curl', '-X', 'POST',
                f"{self.base_url}/api/v1/rag/search/collections/upload",
                '-F', f"file=@{self.test_config['image_path']}",
                '-F', f"query={self.test_config['text_query']}",
                '-F', f"mode={self.test_config['mode']}",
                '-F', f"top_k={self.test_config['top_k']}",
                '-s'  # 静默模式
            ]
            
            # 添加文本集合参数
            for collection in self.test_config['text_collections']:
                curl_cmd.extend(['-F', f'text_collections={collection}'])
            
            # 添加图片集合参数
            for collection in self.test_config['image_collections']:
                curl_cmd.extend(['-F', f'image_collections={collection}'])
            
            print("正在执行curl命令...")
            print(f"命令: {' '.join(curl_cmd[:10])}...")  # 只显示前10个参数
            
            start_time = time.time()
            
            # 执行curl命令
            result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=120)
            
            response_time = time.time() - start_time
            print(f"请求完成，响应时间: {response_time:.3f} 秒")
            print(f"返回码: {result.returncode}")
            
            if result.returncode == 0:
                try:
                    # 解析JSON响应
                    response_data = json.loads(result.stdout)
                    print("✅ API调用成功！")
                    
                    # 详细分析返回结果
                    return self.analyze_response(response_data, response_time)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON解析失败: {e}")
                    print(f"原始响应: {result.stdout[:500]}...")
                    return {
                        'status': 'error',
                        'error_message': f"JSON解析失败: {e}",
                        'raw_response': result.stdout,
                        'response_time': response_time
                    }
            else:
                print(f"❌ curl命令执行失败")
                print(f"错误输出: {result.stderr}")
                print(f"标准输出: {result.stdout}")
                return {
                    'status': 'error',
                    'error_message': f"curl命令失败: {result.stderr}",
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'response_time': response_time
                }
                    
        except Exception as e:
            print(f"❌ 请求过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
    
    def analyze_response(self, result: Dict[str, Any], response_time: float) -> Dict[str, Any]:
        """分析API响应结果"""
        print("\n" + "="*60)
        print("API响应结果分析")
        print("="*60)
        
        analysis = {
            'status': 'success',
            'response_time': response_time,
            'raw_response': result,
            'summary': {},
            'detailed_analysis': {}
        }
        
        # 基础信息 - 修复响应结构解析
        data = result.get('data', {})
        results = data.get('results', [])
        
        print("1. 基础信息:")
        print(f"   总结果数量: {len(results)}")
        print(f"   响应时间: {data.get('response_time', response_time):.3f} 秒")
        print(f"   查询模式: {data.get('mode', 'unknown')}")
        print(f"   融合策略: {data.get('fusion_strategy', 'none')}")
        
        analysis['summary'] = {
            'total_results': len(results),
            'response_time': data.get('response_time', response_time),
            'selected_mode': data.get('mode', 'unknown'),
            'fusion_strategy': data.get('fusion_strategy', 'none')
        }
        
        # 输入解析信息
        print("\n2. 输入信息:")
        print(f"   文本查询: {data.get('query', 'None')}")
        print(f"   图片路径: {data.get('image_path', 'None')}")
        print(f"   模式: {data.get('mode', 'unknown')}")
        print(f"   文本集合: {data.get('text_collections', [])}")
        print(f"   图片集合: {data.get('image_collections', [])}")
        
        analysis['detailed_analysis']['input_info'] = {
            'query': data.get('query'),
            'image_path': data.get('image_path'),
            'mode': data.get('mode'),
            'text_collections': data.get('text_collections', []),
            'image_collections': data.get('image_collections', [])
        }
        
        # 元数据信息
        metadata = data.get('metadata', {})
        if metadata:
            print("\n3. 元数据信息:")
            print(f"   总集合数: {metadata.get('total_collections_searched', 0)}")
            print(f"   集合详情: {metadata.get('collection_breakdown', {})}")
            
            # 各集合的检索详情
            results_by_collection = metadata.get('results_by_collection', {})
            if results_by_collection:
                print("\n4. 各集合检索详情:")
                for collection_name, collection_data in results_by_collection.items():
                    print(f"   {collection_name}:")
                    print(f"     集合类型: {collection_data.get('collection_type', 'unknown')}")
                    print(f"     结果数量: {collection_data.get('total_results', 0)}")
                    print(f"     检索组件: {collection_data.get('retrieval_breakdown', {})}")
                    print(f"     组件计数: {collection_data.get('component_counts', {})}")
            
            analysis['detailed_analysis']['metadata'] = metadata
        
        # 结果详情
        if results:
            print(f"\n5. 结果详情 (前5个):")
            for i, res in enumerate(results[:5]):
                print(f"   结果 {i+1}:")
                print(f"     文档ID: {res.get('doc_id', 'unknown')}")
                print(f"     内容预览: {res.get('content', '')[:100]}...")
                print(f"     相似度分数: {res.get('score', 0):.4f}")
                print(f"     检索类型: {res.get('retrieval_type', 'unknown')}")
                print(f"     资源集合: {res.get('resource_collection', 'unknown')}")
                print(f"     元数据: {list(res.get('metadata', {}).keys())}")
                print()
            
            analysis['detailed_analysis']['top_results'] = results[:5]
        
        # 统计信息
        print("\n7. 统计信息:")
        retrieval_types = {}
        collections = {}
        scores = []
        
        for res in results:
            # 检索类型统计
            ret_type = res.get('retrieval_type', 'unknown')
            retrieval_types[ret_type] = retrieval_types.get(ret_type, 0) + 1
            
            # 集合统计
            collection = res.get('resource_collection', 'unknown')
            collections[collection] = collections.get(collection, 0) + 1
            
            # 分数统计
            score = res.get('score', 0)
            if isinstance(score, (int, float)):
                scores.append(score)
        
        print(f"   检索类型分布: {retrieval_types}")
        print(f"   集合分布: {collections}")
        if scores:
            print(f"   分数范围: {min(scores):.4f} - {max(scores):.4f}")
            print(f"   平均分数: {sum(scores)/len(scores):.4f}")
        
        analysis['detailed_analysis']['statistics'] = {
            'retrieval_types': retrieval_types,
            'collections': collections,
            'score_range': [min(scores), max(scores)] if scores else [0, 0],
            'average_score': sum(scores)/len(scores) if scores else 0
        }
        
        return analysis
    
    def save_results(self, analysis: Dict[str, Any]):
        """保存测试结果"""
        output_file = "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/test/auto_mode_api_test_results.json"
        
        # 准备保存的数据
        save_data = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'test_config': self.test_config,
                'test_type': 'auto_mode_api'
            },
            'analysis': analysis
        }
        
        # 转换复杂对象为可序列化格式
        def convert_for_serialization(data):
            if isinstance(data, dict):
                return {k: convert_for_serialization(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [convert_for_serialization(item) for item in data]
            else:
                return data
        
        serializable_data = convert_for_serialization(save_data)
        
        # 保存到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}")
        print(f"测试结果已保存到: {output_file}")
        print(f"{'='*80}")
    
    def run_test(self):
        """运行完整测试"""
        print("="*80)
        print("Auto模式API测试")
        print("="*80)
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 检查服务器状态
        if not self.check_server_status():
            print("❌ 服务器不可用，测试终止")
            return
        
        # 2. 检查图片文件
        if not self.check_image_file():
            print("❌ 图片文件不存在，测试终止")
            return
        
        # 3. 执行API测试
        analysis = self.test_auto_mode_api()
        
        # 4. 保存结果
        self.save_results(analysis)
        
        # 5. 打印总结
        print("\n" + "="*80)
        print("测试总结")
        print("="*80)
        
        if analysis.get('status') == 'success':
            summary = analysis.get('summary', {})
            print(f"✅ 测试成功完成")
            print(f"   总结果数量: {summary.get('total_results', 0)}")
            print(f"   响应时间: {summary.get('response_time', 0):.3f} 秒")
            print(f"   查询模式: {summary.get('selected_mode', 'unknown')}")
            print(f"   融合策略: {summary.get('fusion_strategy', 'none')}")
        else:
            print(f"❌ 测试失败")
            print(f"   错误信息: {analysis.get('error_message', 'unknown')}")
        
        print("="*80)


def main():
    """主函数"""
    try:
        tester = AutoModeAPITester()
        tester.run_test()
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()