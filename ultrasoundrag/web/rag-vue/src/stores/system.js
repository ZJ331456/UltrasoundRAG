// 系统状态管理
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { api } from '../api/index.js'

export const useSystemStore = defineStore('system', () => {
  // 状态
  const systemStatus = ref({
    status: 'unknown',
    version: '3.0.0',
    timestamp: null,
    details: {}
  })
  
  const databases = ref([])
  const alerts = ref([])
  const loading = ref(false)
  const error = ref(null)

  // 计算属性
  const isHealthy = computed(() => {
    const status = systemStatus.value.status
    return status === 'healthy' || status === 'running'
  })
  const alertCount = computed(() => alerts.value.filter(alert => alert.active).length)

  // 方法
  const checkSystemHealth = async () => {
    try {
      loading.value = true
      error.value = null
      
      const response = await api.get('/health/')
      systemStatus.value = response.data
      
      return response.data
    } catch (err) {
      if (err.code === 'ECONNABORTED') {
        error.value = '健康检查超时，系统可能正在处理大型任务'
        systemStatus.value = {
          status: 'timeout',
          version: '3.0.0',
          timestamp: Date.now() / 1000,
          details: { note: '请求超时，可能正在执行长时间任务' }
        }
      } else {
        error.value = err.message || '健康检查失败'
        systemStatus.value = {
          status: 'error',
          version: '3.0.0',
          timestamp: Date.now() / 1000,
          details: { error: err.message }
        }
      }
      console.error('健康检查失败:', err)
      throw err
    } finally {
      loading.value = false
    }
  }

  const getSystemStatus = async () => {
    try {
      loading.value = true
      const response = await api.get('/health/system/status')
      
      if (response.data.success) {
        systemStatus.value = response.data.data
      }
      
      return response.data
    } catch (err) {
      error.value = err.message || '获取系统状态失败'
      throw err
    } finally {
      loading.value = false
    }
  }

  const loadDatabases = async () => {
    try {
      const response = await api.get('/databases')
      
      if (response.data.success) {
        databases.value = response.data.data?.databases || []
      }
      
      return response.data
    } catch (err) {
      error.value = err.message || '加载数据库列表失败'
      throw err
    }
  }

  const reloadConfig = async () => {
    try {
      const response = await api.post('/system/reload-config')
      return response.data
    } catch (err) {
      error.value = err.message || '重新加载配置失败'
      throw err
    }
  }

  // 返回状态和方法
  return {
    // 状态
    systemStatus,
    databases,
    alerts,
    loading,
    error,
    
    // 计算属性
    isHealthy,
    alertCount,
    
    // 方法
    checkSystemHealth,
    getSystemStatus,
    loadDatabases,
    reloadConfig
  }
})
