#!/bin/bash

# 强制重建前端，清除所有缓存

echo "🧹 强制重建前端应用并清除缓存..."

cd /media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/ultrasoundrag/web/rag-vue

echo "1️⃣ 停止开发服务器 (如果在运行)"
# 这里会让用户手动停止，因为我们无法控制其他终端

echo "2️⃣ 清除node_modules和lock文件"
rm -rf node_modules
rm -f package-lock.json

echo "3️⃣ 清除构建缓存"
rm -rf dist
rm -rf .vite

echo "4️⃣ 重新安装依赖"
npm install

echo "5️⃣ 重新构建应用"
npm run build

echo "6️⃣ 构建完成"
ls -la dist/

echo ""
echo "✅ 前端重建完成！"
echo ""
echo "📋 接下来的步骤："
echo "1. 重启后端API服务器"
echo "2. 在浏览器中按 Ctrl+Shift+R 强制刷新"
echo "3. 检查Network面板确认API调用路径正确"
echo ""
