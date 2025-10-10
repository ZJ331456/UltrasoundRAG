// 任务追踪状态管理
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { taskAPI } from '../api/index.js'
import { message } from 'ant-design-vue'

export const useTaskStore = defineStore('task', () => {
  // 状态
  const tasks = ref([])
  const activeTasks = ref(new Map()) // 当前活跃任务的追踪器
  const loading = ref(false)
  const error = ref(null)

  // 计算属性
  const runningTasks = computed(() => {
    return tasks.value.filter(task => task.status === 'running')
  })

  const completedTasks = computed(() => {
    return tasks.value.filter(task => task.status === 'completed')
  })

  const failedTasks = computed(() => {
    return tasks.value.filter(task => task.status === 'failed')
  })

  const taskCount = computed(() => tasks.value.length)
  const hasRunningTasks = computed(() => runningTasks.value.length > 0)

  // 方法
  const loadTasks = async () => {
    try {
      loading.value = true
      error.value = null
      
      const response = await taskAPI.list()
      
      if (response.data.success) {
        tasks.value = response.data.data?.tasks || []
      } else {
        throw new Error(response.data.error || '加载任务列表失败')
      }
      
      return response.data
    } catch (err) {
      error.value = err.message || '加载任务列表失败'
      console.error('加载任务失败:', err)
      throw err
    } finally {
      loading.value = false
    }
  }

  const getTaskStatus = async (taskId) => {
    try {
      const response = await taskAPI.get(taskId)
      
      if (response.data.success) {
        const task = response.data.data
        
        // 更新本地任务列表中的对应任务
        const index = tasks.value.findIndex(t => t.id === taskId)
        if (index !== -1) {
          tasks.value[index] = task
        } else {
          tasks.value.push(task)
        }
        
        return task
      } else {
        throw new Error(response.data.error || '获取任务状态失败')
      }
    } catch (err) {
      error.value = err.message || '获取任务状态失败'
      console.error('获取任务状态失败:', err)
      throw err
    }
  }

  const deleteTask = async (taskId) => {
    try {
      const response = await taskAPI.delete(taskId)
      
      if (response.data.success) {
        // 从本地列表中移除
        tasks.value = tasks.value.filter(t => t.id !== taskId)
        
        // 停止追踪
        stopTaskTracking(taskId)
        
        message.success('任务删除成功')
        return response.data
      } else {
        throw new Error(response.data.error || '删除任务失败')
      }
    } catch (err) {
      error.value = err.message || '删除任务失败'
      message.error(error.value)
      throw err
    }
  }

  const startTaskTracking = (taskId, onUpdate = null, onComplete = null, onError = null) => {
    // 如果已经在追踪，先停止
    if (activeTasks.value.has(taskId)) {
      stopTaskTracking(taskId)
    }

    const intervalId = setInterval(async () => {
      try {
        const task = await getTaskStatus(taskId)
        
        // 调用更新回调
        if (onUpdate) {
          onUpdate(task)
        }
        
        // 如果任务完成，停止追踪
        if (task.status === 'completed') {
          stopTaskTracking(taskId)
          if (onComplete) {
            onComplete(task)
          }
          message.success(`任务 "${task.name}" 完成`)
        } else if (task.status === 'failed') {
          stopTaskTracking(taskId)
          if (onError) {
            onError(task)
          }
          message.error(`任务 "${task.name}" 失败: ${task.error || '未知错误'}`)
        }
      } catch (err) {
        console.error('追踪任务状态失败:', err)
        // 如果获取状态失败，可能任务已被删除或服务不可用，停止追踪
        stopTaskTracking(taskId)
        if (onError) {
          onError({ error: err.message })
        }
      }
    }, 5000) // 每5秒检查一次

    activeTasks.value.set(taskId, {
      intervalId,
      onUpdate,
      onComplete,
      onError
    })
  }

  const stopTaskTracking = (taskId) => {
    const tracker = activeTasks.value.get(taskId)
    if (tracker) {
      clearInterval(tracker.intervalId)
      activeTasks.value.delete(taskId)
    }
  }

  const stopAllTracking = () => {
    activeTasks.value.forEach((tracker, taskId) => {
      clearInterval(tracker.intervalId)
    })
    activeTasks.value.clear()
  }

  const isTaskTracking = (taskId) => {
    return activeTasks.value.has(taskId)
  }

  const trackTaskUntilComplete = (taskId) => {
    return new Promise((resolve, reject) => {
      startTaskTracking(
        taskId,
        null, // onUpdate
        (task) => resolve(task), // onComplete
        (error) => reject(error) // onError
      )
    })
  }

  // 清除错误
  const clearError = () => {
    error.value = null
  }

  // 清除已完成的任务
  const clearCompletedTasks = () => {
    const completedTaskIds = completedTasks.value.map(t => t.id)
    tasks.value = tasks.value.filter(t => !completedTaskIds.includes(t.id))
    
    // 停止对已完成任务的追踪
    completedTaskIds.forEach(id => stopTaskTracking(id))
    
    message.success(`已清除 ${completedTaskIds.length} 个已完成的任务`)
  }

  // 返回状态和方法
  return {
    // 状态
    tasks,
    activeTasks,
    loading,
    error,
    
    // 计算属性
    runningTasks,
    completedTasks,
    failedTasks,
    taskCount,
    hasRunningTasks,
    
    // 方法
    loadTasks,
    getTaskStatus,
    deleteTask,
    startTaskTracking,
    stopTaskTracking,
    stopAllTracking,
    isTaskTracking,
    trackTaskUntilComplete,
    clearError,
    clearCompletedTasks
  }
})
