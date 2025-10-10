<template>
  <div class="dashboard">
    <!-- 页面标题 -->
    <a-page-header
      title="系统仪表盘"
      sub-title="UltrasoundRAG 系统概览"
      style="padding: 0 0 24px 0;"
    />

    <!-- 关键指标卡片 -->
    <a-row :gutter="[16, 16]" style="margin-bottom: 24px;">
      <a-col :xs="24" :sm="12" :md="6">
        <a-card class="metric-card">
          <a-statistic
            title="系统状态"
            :value="systemStatus.status"
            :value-style="{ color: isHealthy ? '#3f8600' : '#cf1322' }"
          >
            <template #formatter="{ value }">
              <a-badge 
                :status="isHealthy ? 'success' : 'error'" 
                :text="value === 'healthy' ? '正常运行' : '异常'"
              />
            </template>
          </a-statistic>
        </a-card>
      </a-col>
      
      <a-col :xs="24" :sm="12" :md="6">
        <a-card class="metric-card">
          <a-statistic
            title="数据库数量"
            :value="databaseCount"
            :value-style="{ color: '#1890ff' }"
            suffix="个"
          />
        </a-card>
      </a-col>
      
      <a-col :xs="24" :sm="12" :md="6">
        <a-card class="metric-card">
          <a-statistic
            title="活跃告警"
            :value="alertCount"
            :value-style="{ color: alertCount > 0 ? '#cf1322' : '#3f8600' }"
            suffix="条"
          />
        </a-card>
      </a-col>
      
      <a-col :xs="24" :sm="12" :md="6">
        <a-card class="metric-card">
          <a-statistic
            title="系统版本"
            value="v1.0.0"
            :value-style="{ color: '#722ed1' }"
          />
        </a-card>
      </a-col>
    </a-row>

    <a-row :gutter="[16, 16]">
      <!-- 快速操作 -->
      <a-col :xs="24" :lg="12">
        <a-card title="快速操作" :bordered="false">
          <a-space direction="vertical" size="middle" style="width: 100%;">
            <a-button 
              type="primary" 
              size="large" 
              block
              @click="$router.push('/app/search')"
            >
              <SearchOutlined />
              开始智能检索
            </a-button>
            
            <a-button 
              size="large" 
              block
              @click="$router.push('/app/collection')"
            >
              <FolderOpenOutlined />
              管理集合
            </a-button>
            
            <a-button 
              size="large" 
              block
              @click="$router.push('/app/database')"
            >
              <DatabaseOutlined />
              管理数据库
            </a-button>
            
            <a-button 
              size="large" 
              block
              @click="buildIndexes"
              :loading="buildingIndexes"
            >
              <ReloadOutlined />
              重建索引
            </a-button>
            
            <a-button 
              size="large" 
              block
              @click="runSystemTest"
              :loading="runningTest"
            >
              <ExperimentOutlined />
              运行系统测试
            </a-button>
          </a-space>
        </a-card>
      </a-col>

      <!-- 系统信息 -->
      <a-col :xs="24" :lg="12">
        <a-card title="系统信息" :bordered="false">
          <a-descriptions :column="1" size="small">
            <a-descriptions-item label="运行时间">
              {{ uptime }}
            </a-descriptions-item>
            <a-descriptions-item label="最后检查">
              {{ lastCheckTime }}
            </a-descriptions-item>
            <a-descriptions-item label="API端点">
              <a-tag color="blue">{{ apiEndpoint }}</a-tag>
            </a-descriptions-item>
            <a-descriptions-item label="前端版本">
              <a-tag color="green">Vue 3.5.18</a-tag>
            </a-descriptions-item>
          </a-descriptions>
        </a-card>
      </a-col>
    </a-row>

    <!-- 最近活动 -->
    <a-card title="最近活动" style="margin-top: 24px;" :bordered="false">
      <a-timeline>
        <a-timeline-item v-for="activity in recentActivities" :key="activity.id">
          <template #dot>
            <component :is="activity.icon" style="font-size: 16px;" />
          </template>
          <div>
            <div style="font-weight: 500;">{{ activity.title }}</div>
            <div style="color: rgba(0, 0, 0, 0.45); font-size: 12px;">
              {{ activity.time }}
            </div>
          </div>
        </a-timeline-item>
      </a-timeline>
    </a-card>
    
    <!-- 任务进度追踪组件 -->
    <TaskProgress ref="taskProgressRef" @task-complete="handleTaskComplete" />
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { message } from 'ant-design-vue'
import {
  SearchOutlined,
  DatabaseOutlined,
  FolderOpenOutlined,
  ReloadOutlined,
  ExperimentOutlined,
  CheckCircleOutlined,
  InfoCircleOutlined,
  WarningOutlined
} from '@ant-design/icons-vue'
import { useSystemStore } from '../stores/system'
import { useTaskStore } from '../stores/task'
import { systemAPI } from '../api/index.js'
import TaskProgress from '../components/TaskProgress.vue'
import dayjs from 'dayjs'

// 状态管理
const systemStore = useSystemStore()
const taskStore = useTaskStore()

// 响应式数据
const buildingIndexes = ref(false)
const runningTest = ref(false)
const currentTime = ref(dayjs())
const taskProgressRef = ref(null)


// 计算属性
const systemStatus = computed(() => systemStore.systemStatus)
const isHealthy = computed(() => systemStore.isHealthy)
const alertCount = computed(() => systemStore.alertCount)
const databaseCount = computed(() => systemStore.databases.length)

const uptime = computed(() => {
  const startTime = systemStatus.value.timestamp
  if (!startTime) return '未知'
  
  const duration = dayjs().diff(dayjs(startTime * 1000), 'minute')
  if (duration < 60) return `${duration} 分钟`
  if (duration < 1440) return `${Math.floor(duration / 60)} 小时`
  return `${Math.floor(duration / 1440)} 天`
})

const lastCheckTime = computed(() => {
  return currentTime.value.format('YYYY-MM-DD HH:mm:ss')
})

const apiEndpoint = computed(() => window.location.origin + '/api/v1/rag')

const recentActivities = ref([
  {
    id: 1,
    title: '系统健康检查完成',
    time: '刚刚',
    icon: CheckCircleOutlined
  },
  {
    id: 2,
    title: '数据库连接正常',
    time: '2 分钟前',
    icon: InfoCircleOutlined
  },
  {
    id: 3,
    title: '检索服务启动',
    time: '5 分钟前',
    icon: CheckCircleOutlined
  }
])

// 方法
const buildIndexes = async () => {
  try {
    buildingIndexes.value = true
    
    const response = await systemAPI.buildIndexes({
      target: 'all',
      recreate: true
    })
    
    if (response.data.success) {
      const taskId = response.data.task_id
      message.success('索引构建任务已启动！')
      
      // 添加到活动记录
      recentActivities.value.unshift({
        id: Date.now(),
        title: '索引重建任务已启动',
        time: '刚刚',
        icon: ReloadOutlined
      })
      
      // 显示任务进度追踪
      if (taskProgressRef.value) {
        taskProgressRef.value.showProgress(taskId)
      }
      
    } else {
      throw new Error(response.data.message || '启动索引构建任务失败')
    }
  } catch (error) {
    console.error('启动索引构建失败:', error)
    message.error(`启动索引构建失败: ${error.message}`)
  } finally {
    buildingIndexes.value = false
  }
}

const handleTaskComplete = (task) => {
  // 任务完成时的处理
  if (task.name.includes('索引')) {
    recentActivities.value.unshift({
      id: Date.now(),
      title: `${task.name} 已完成`,
      time: '刚刚',
      icon: CheckCircleOutlined
    })
    
    // 刷新系统状态
    refreshData()
  }
}

const runSystemTest = async () => {
  try {
    runningTest.value = true
    
    const response = await systemAPI.runTests({
      test_type: 'retrieval',
      params: {}
    })
    
    if (response.data.success) {
      message.success('系统测试运行完成')
      
      // 添加到活动记录
      recentActivities.value.unshift({
        id: Date.now(),
        title: '系统测试完成',
        time: '刚刚',
        icon: ExperimentOutlined
      })
    } else {
      throw new Error(response.data.error || '系统测试失败')
    }
  } catch (error) {
    console.error('运行测试失败:', error)
  } finally {
    runningTest.value = false
  }
}

const refreshData = async () => {
  try {
    await Promise.all([
      systemStore.checkSystemHealth(),
      systemStore.loadDatabases()
    ])
    currentTime.value = dayjs()
  } catch (error) {
    console.error('刷新数据失败:', error)
  }
}

// 定时器
let refreshTimer = null

// 生命周期
onMounted(async () => {
  await refreshData()
  
  // 每2分钟刷新一次数据
  refreshTimer = setInterval(refreshData, 120000)
})

onUnmounted(() => {
  if (refreshTimer) {
    clearInterval(refreshTimer)
  }
})
</script>

<style scoped>
.dashboard {
  max-width: 1200px;
  margin: 0 auto;
}

.metric-card {
  text-align: center;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
}

.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
}

:deep(.ant-statistic-content) {
  font-size: 24px;
  font-weight: bold;
}

:deep(.ant-card-head-title) {
  font-size: 16px;
  font-weight: 600;
}

:deep(.ant-timeline-item-content) {
  margin-left: 16px;
}
</style>
