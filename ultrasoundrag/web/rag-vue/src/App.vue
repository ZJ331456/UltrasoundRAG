<template>
  <a-layout class="app-layout">
    <!-- 顶部导航栏 -->
    <a-layout-header class="app-header">
      <div class="logo">
        🩺 UltrasoundRAG v3.0
      </div>
      <div class="user-info">
        <a-space>
          <a-badge :count="alertCount" :offset="[10, 0]">
            <BellOutlined style="font-size: 18px; color: white;" />
          </a-badge>
          <a-avatar :size="32" style="background-color: rgba(255, 255, 255, 0.2)">
            <UserOutlined />
          </a-avatar>
          <span>系统管理员</span>
        </a-space>
      </div>
    </a-layout-header>

    <a-layout>
      <!-- 侧边栏 -->
      <a-layout-sider 
        v-model:collapsed="collapsed" 
        :trigger="null" 
        collapsible 
        width="200"
        class="app-sider"
      >
        <a-menu
          v-model:selectedKeys="selectedKeys"
          mode="inline"
          :style="{ height: '100%', borderRight: 0 }"
          @click="handleMenuClick"
        >
          <a-menu-item key="dashboard">
            <DashboardOutlined />
            <span>仪表盘</span>
          </a-menu-item>
          <a-menu-item key="search">
            <SearchOutlined />
            <span>智能检索</span>
          </a-menu-item>
          
          <a-menu-divider />
          
          <a-menu-item-group title="数据管理">
            <a-menu-item key="data">
              <DatabaseOutlined />
              <span>数据管理</span>
            </a-menu-item>
          </a-menu-item-group>
          
          <a-menu-divider />
          
          <a-menu-item key="status">
            <MonitorOutlined />
            <span>系统状态</span>
          </a-menu-item>
        </a-menu>
      </a-layout-sider>

      <!-- 主内容区域 -->
      <a-layout>
        <a-layout-content class="app-content">
          <!-- 面包屑导航 -->
          <a-breadcrumb style="margin-bottom: 16px;">
            <a-breadcrumb-item>
              <HomeOutlined />
              <span>首页</span>
            </a-breadcrumb-item>
            <a-breadcrumb-item>{{ currentPageTitle }}</a-breadcrumb-item>
          </a-breadcrumb>

          <!-- 路由视图 -->
          <router-view v-slot="{ Component }">
            <transition name="fade" mode="out-in">
              <component :is="Component" />
            </transition>
          </router-view>
        </a-layout-content>
      </a-layout>
    </a-layout>

    <!-- 全局加载遮罩 -->
    <a-spin 
      :spinning="globalLoading" 
      tip="正在处理..." 
      size="large"
      style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 9999; background: rgba(255, 255, 255, 0.8);"
    />
  </a-layout>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { 
  DashboardOutlined, 
  SearchOutlined, 
  DatabaseOutlined, 
  FolderOpenOutlined,
  FileTextOutlined,
  MonitorOutlined,
  UserOutlined,
  BellOutlined,
  HomeOutlined
} from '@ant-design/icons-vue'
import { message } from 'ant-design-vue'
import { useSystemStore } from './stores/system'

// 路由
const router = useRouter()
const route = useRoute()

// 状态管理
const systemStore = useSystemStore()

// 响应式数据
const collapsed = ref(false)
const selectedKeys = ref(['dashboard'])
const globalLoading = ref(false)

// 计算属性
const alertCount = computed(() => systemStore.alertCount)
const currentPageTitle = computed(() => {
  const titles = {
    'dashboard': '仪表盘',
    'search': '智能检索',
    'data': '数据管理',
    'status': '系统状态'
  }
  return titles[selectedKeys.value[0]] || '首页'
})

// 方法
const handleMenuClick = ({ key }) => {
  selectedKeys.value = [key]
  router.push(`/app/${key}`)
}

// 监听路由变化
watch(() => route.path, (newPath) => {
  const pathSegments = newPath.split('/')
  const currentPage = pathSegments[pathSegments.length - 1]
  if (['dashboard', 'search', 'data', 'status'].includes(currentPage)) {
    selectedKeys.value = [currentPage]
  }
}, { immediate: true })

// 初始化
onMounted(async () => {
  try {
    globalLoading.value = true
    
    // 检查系统状态
    await systemStore.checkSystemHealth()
    
    // 显示欢迎消息
    message.success('欢迎使用 UltrasoundRAG 系统！', 3)
    
  } catch (error) {
    console.error('应用初始化失败:', error)
    message.error('系统初始化失败，请检查后端服务是否正常运行')
  } finally {
    globalLoading.value = false
  }
})

// 暴露给模板的数据和方法
defineExpose({
  globalLoading,
  collapsed,
  selectedKeys,
  handleMenuClick
})
</script>

<style scoped>
.app-layout {
  min-height: 100vh;
}

.app-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 0;
  line-height: 64px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  position: relative;
  z-index: 1;
}

.logo {
  float: left;
  width: 200px;
  height: 64px;
  line-height: 64px;
  padding-left: 24px;
  color: white;
  font-size: 20px;
  font-weight: bold;
  user-select: none;
}

.user-info {
  float: right;
  margin-right: 24px;
  color: white;
  line-height: 64px;
}

.app-sider {
  background: #fff;
  box-shadow: 2px 0 8px rgba(0, 0, 0, 0.1);
  z-index: 2;
}

.app-content {
  margin: 24px;
  padding: 24px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  min-height: calc(100vh - 112px);
}

/* 过渡动画 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .logo {
    width: 150px;
    font-size: 16px;
  }
  
  .user-info {
    margin-right: 16px;
  }
  
  .app-content {
    margin: 16px;
    padding: 16px;
  }
}
</style>
