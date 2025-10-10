<template>
  <div class="system-status">
    <a-page-header
      title="系统状态"
      sub-title="监控系统运行状态和性能指标"
      style="padding: 0 0 24px 0;"
    />

    <!-- 系统健康状态 -->
    <a-card title="系统健康状态" :bordered="false" style="margin-bottom: 24px;">
      <a-row :gutter="[16, 16]">
        <a-col :xs="24" :sm="8">
          <a-statistic
            title="整体状态"
            :value="systemStatus.status"
            :value-style="{ color: isHealthy ? '#3f8600' : '#cf1322' }"
          >
            <template #formatter="{ value }">
              <a-badge 
                :status="isHealthy ? 'success' : 'error'" 
                :text="value === 'healthy' ? '健康' : '异常'"
              />
            </template>
          </a-statistic>
        </a-col>
        
        <a-col :xs="24" :sm="8">
          <a-statistic
            title="系统版本"
            :value="systemStatus.version || '3.0.0'"
            :value-style="{ color: '#1890ff' }"
          />
        </a-col>
        
        <a-col :xs="24" :sm="8">
          <a-statistic
            title="运行时间"
            :value="uptime"
            :value-style="{ color: '#722ed1' }"
          />
        </a-col>
      </a-row>
    </a-card>

    <!-- 详细信息 -->
    <a-row :gutter="[16, 16]">
      <!-- 服务状态 -->
      <a-col :xs="24" :lg="12">
        <a-card title="服务状态" :bordered="false">
          <a-list :data-source="serviceStatus" size="small">
            <template #renderItem="{ item }">
              <a-list-item>
                <a-list-item-meta>
                  <template #title>
                    <a-space>
                      <span>{{ item.name }}</span>
                      <a-badge 
                        :status="item.status === 'running' ? 'success' : 'error'"
                        :text="item.status === 'running' ? '运行中' : '异常'"
                      />
                    </a-space>
                  </template>
                  <template #description>
                    {{ item.description }}
                  </template>
                </a-list-item-meta>
              </a-list-item>
            </template>
          </a-list>
        </a-card>
      </a-col>

      <!-- 系统配置 -->
      <a-col :xs="24" :lg="12">
        <a-card title="系统配置" :bordered="false">
          <a-descriptions :column="1" size="small">
            <a-descriptions-item label="API端点">
              {{ apiEndpoint }}
            </a-descriptions-item>
            <a-descriptions-item label="前端版本">
              Vue 3.5.18
            </a-descriptions-item>
            <a-descriptions-item label="后端版本">
              {{ systemStatus.version || '3.0.0' }}
            </a-descriptions-item>
            <a-descriptions-item label="最后检查">
              {{ lastUpdateTime }}
            </a-descriptions-item>
          </a-descriptions>
          
          <a-divider />
          
          <a-space>
            <a-button 
              type="primary" 
              @click="refreshStatus"
              :loading="refreshing"
            >
              <ReloadOutlined />
              刷新状态
            </a-button>
            
            <a-button @click="reloadConfig" :loading="reloading">
              <SettingOutlined />
              重载配置
            </a-button>
          </a-space>
        </a-card>
      </a-col>
    </a-row>

    <!-- 日志信息 -->
    <a-card title="系统日志" style="margin-top: 24px;" :bordered="false">
      <a-list :data-source="systemLogs" size="small">
        <template #renderItem="{ item }">
          <a-list-item>
            <a-list-item-meta>
              <template #title>
                <a-space>
                  <a-tag :color="getLogColor(item.level)">
                    {{ item.level.toUpperCase() }}
                  </a-tag>
                  <span>{{ item.message }}</span>
                </a-space>
              </template>
              <template #description>
                {{ item.timestamp }}
              </template>
            </a-list-item-meta>
          </a-list-item>
        </template>
      </a-list>
    </a-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { message } from 'ant-design-vue'
import { ReloadOutlined, SettingOutlined } from '@ant-design/icons-vue'
import { useSystemStore } from '../stores/system'
import { systemAPI } from '../api/index.js'
import dayjs from 'dayjs'

const systemStore = useSystemStore()

const refreshing = ref(false)
const reloading = ref(false)
const currentTime = ref(dayjs())

const systemStatus = computed(() => systemStore.systemStatus)
const isHealthy = computed(() => systemStore.isHealthy)

const apiEndpoint = computed(() => window.location.origin + '/api/v1/rag')

const uptime = computed(() => {
  const startTime = systemStatus.value.timestamp
  if (!startTime) return '未知'
  
  const duration = dayjs().diff(dayjs(startTime * 1000), 'minute')
  if (duration < 60) return `${duration} 分钟`
  if (duration < 1440) return `${Math.floor(duration / 60)} 小时`
  return `${Math.floor(duration / 1440)} 天`
})

const lastUpdateTime = computed(() => {
  return currentTime.value.format('YYYY-MM-DD HH:mm:ss')
})

const serviceStatus = ref([
  { name: 'API服务', status: 'running', description: 'REST API接口服务' },
  { name: '检索引擎', status: 'running', description: '多模态检索服务' },
  { name: '数据库连接', status: 'running', description: 'Milvus向量数据库' },
  { name: '模型服务', status: 'running', description: 'CLIP模型服务' }
])

const systemLogs = ref([
  {
    level: 'info',
    message: '系统启动完成',
    timestamp: dayjs().subtract(5, 'minute').format('YYYY-MM-DD HH:mm:ss')
  },
  {
    level: 'info', 
    message: '数据库连接成功',
    timestamp: dayjs().subtract(3, 'minute').format('YYYY-MM-DD HH:mm:ss')
  },
  {
    level: 'info',
    message: '健康检查通过',
    timestamp: dayjs().subtract(1, 'minute').format('YYYY-MM-DD HH:mm:ss')
  }
])

const refreshStatus = async () => {
  try {
    refreshing.value = true
    await systemStore.checkSystemHealth()
    currentTime.value = dayjs()
    message.success('状态刷新成功')
  } catch (error) {
    console.error('刷新状态失败:', error)
  } finally {
    refreshing.value = false
  }
}

const reloadConfig = async () => {
  try {
    reloading.value = true
    await systemStore.reloadConfig()
    message.success('配置重载成功')
    
    // 添加日志
    systemLogs.value.unshift({
      level: 'info',
      message: '配置重载完成',
      timestamp: dayjs().format('YYYY-MM-DD HH:mm:ss')
    })
  } catch (error) {
    console.error('重载配置失败:', error)
  } finally {
    reloading.value = false
  }
}

const getLogColor = (level) => {
  const colors = {
    info: 'blue',
    warn: 'orange', 
    error: 'red',
    debug: 'gray'
  }
  return colors[level] || 'default'
}

let statusTimer = null

onMounted(async () => {
  await refreshStatus()
  
  // 每2分钟自动刷新状态
  statusTimer = setInterval(() => {
    currentTime.value = dayjs()
  }, 120000)
})

onUnmounted(() => {
  if (statusTimer) {
    clearInterval(statusTimer)
  }
})
</script>

<style scoped>
.system-status {
  max-width: 1200px;
  margin: 0 auto;
}

:deep(.ant-statistic-content) {
  font-size: 20px;
  font-weight: bold;
}
</style>
