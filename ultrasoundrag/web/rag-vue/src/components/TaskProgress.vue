<template>
  <div class="task-progress">
    <!-- 任务进度弹窗 -->
    <a-modal
      v-model:open="showModal"
      :title="modalTitle"
      :closable="!isRunning"
      :maskClosable="!isRunning"
      :footer="null"
      width="600px"
      @cancel="handleCancel"
    >
      <div v-if="currentTask">
        <!-- 任务信息 -->
        <a-descriptions :column="1" bordered size="small" style="margin-bottom: 16px;">
          <a-descriptions-item label="任务名称">
            {{ currentTask.name }}
          </a-descriptions-item>
          <a-descriptions-item label="任务状态">
            <a-badge 
              :status="getStatusBadge(currentTask.status)"
              :text="getStatusText(currentTask.status)"
            />
          </a-descriptions-item>
          <a-descriptions-item label="创建时间">
            {{ formatTime(currentTask.created_at) }}
          </a-descriptions-item>
          <a-descriptions-item label="最后更新">
            {{ formatTime(currentTask.updated_at) }}
          </a-descriptions-item>
        </a-descriptions>

        <!-- 进度条 -->
        <div style="margin-bottom: 16px;">
          <div style="margin-bottom: 8px; display: flex; justify-content: space-between;">
            <span>任务进度</span>
            <span>{{ currentTask.progress.toFixed(1) }}%</span>
          </div>
          <a-progress 
            :percent="currentTask.progress" 
            :status="getProgressStatus(currentTask.status)"
            :show-info="false"
          />
        </div>

        <!-- 当前消息 -->
        <div style="margin-bottom: 16px;">
          <div style="margin-bottom: 8px; font-weight: 500;">当前状态</div>
          <div style="padding: 12px; background: #f5f5f5; border-radius: 4px; min-height: 40px;">
            <a-spin :spinning="isRunning && !currentTask.message" size="small">
              {{ currentTask.message || (isRunning ? '正在处理...' : '等待中...') }}
            </a-spin>
          </div>
        </div>

        <!-- 错误信息 -->
        <div v-if="currentTask.error" style="margin-bottom: 16px;">
          <div style="margin-bottom: 8px; font-weight: 500; color: #ff4d4f;">错误信息</div>
          <div style="padding: 12px; background: #fff2f0; border: 1px solid #ffccc7; border-radius: 4px;">
            {{ currentTask.error }}
          </div>
        </div>

        <!-- 结果信息 -->
        <div v-if="currentTask.result && currentTask.status === 'completed'" style="margin-bottom: 16px;">
          <div style="margin-bottom: 8px; font-weight: 500; color: #52c41a;">执行结果</div>
          <div style="padding: 12px; background: #f6ffed; border: 1px solid #b7eb8f; border-radius: 4px; max-height: 200px; overflow-y: auto;">
            <pre style="margin: 0; white-space: pre-wrap;">{{ JSON.stringify(currentTask.result, null, 2) }}</pre>
          </div>
        </div>

        <!-- 操作按钮 -->
        <div style="text-align: right;">
          <a-space>
            <a-button 
              v-if="!isRunning" 
              @click="handleClose"
            >
              关闭
            </a-button>
            <a-button 
              v-if="!isRunning" 
              type="primary" 
              danger
              @click="handleDelete"
            >
              删除任务
            </a-button>
            <a-button 
              v-if="isRunning" 
              @click="handleMinimize"
            >
              最小化
            </a-button>
          </a-space>
        </div>
      </div>
    </a-modal>

    <!-- 浮动任务列表（最小化时显示） -->
    <div 
      v-if="!showModal && hasRunningTasks" 
      class="floating-tasks"
      @click="showTaskList"
    >
      <a-badge :count="runningTaskCount" :offset="[10, 0]">
        <div class="floating-icon">
          <LoadingOutlined spin />
          <span style="margin-left: 8px;">{{ runningTaskCount }} 个任务运行中</span>
        </div>
      </a-badge>
    </div>

    <!-- 任务列表抽屉 -->
    <a-drawer
      v-model:open="showDrawer"
      title="任务管理"
      placement="right"
      width="400"
    >
      <div class="task-list">
        <div style="margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
          <span>全部任务 ({{ taskCount }})</span>
          <a-space>
            <a-button size="small" @click="refreshTasks">
              <ReloadOutlined />
            </a-button>
            <a-button size="small" @click="clearCompleted" :disabled="completedTaskCount === 0">
              清除已完成
            </a-button>
          </a-space>
        </div>

        <div v-if="taskCount === 0" style="text-align: center; color: #999; padding: 40px 0;">
          暂无任务
        </div>

        <div v-else>
          <div 
            v-for="task in tasks" 
            :key="task.id"
            class="task-item"
            @click="showTaskDetails(task)"
          >
            <div class="task-header">
              <div class="task-name">{{ task.name }}</div>
              <a-badge 
                :status="getStatusBadge(task.status)"
                :text="getStatusText(task.status)"
              />
            </div>
            
            <div class="task-progress-mini">
              <a-progress 
                :percent="task.progress" 
                :status="getProgressStatus(task.status)"
                size="small"
                :show-info="false"
              />
              <span class="progress-text">{{ task.progress.toFixed(1) }}%</span>
            </div>
            
            <div class="task-message">
              {{ task.message || '等待中...' }}
            </div>
            
            <div class="task-time">
              {{ formatTime(task.updated_at) }}
            </div>
          </div>
        </div>
      </div>
    </a-drawer>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { message } from 'ant-design-vue'
import {
  LoadingOutlined,
  ReloadOutlined
} from '@ant-design/icons-vue'
import { useTaskStore } from '../stores/task'
import dayjs from 'dayjs'

// Props
const props = defineProps({
  taskId: {
    type: String,
    default: ''
  },
  autoShow: {
    type: Boolean,
    default: true
  }
})

// Emits
const emit = defineEmits(['taskComplete', 'taskError', 'taskUpdate'])

// 状态管理
const taskStore = useTaskStore()

// 响应式数据
const showModal = ref(false)
const showDrawer = ref(false)
const currentTask = ref(null)

// 计算属性
const tasks = computed(() => taskStore.tasks)
const runningTasks = computed(() => taskStore.runningTasks)
const completedTasks = computed(() => taskStore.completedTasks)
const taskCount = computed(() => taskStore.taskCount)
const runningTaskCount = computed(() => runningTasks.value.length)
const completedTaskCount = computed(() => completedTasks.value.length)
const hasRunningTasks = computed(() => taskStore.hasRunningTasks)

const modalTitle = computed(() => {
  if (!currentTask.value) return '任务详情'
  return `任务详情 - ${currentTask.value.name}`
})

const isRunning = computed(() => {
  return currentTask.value?.status === 'running' || currentTask.value?.status === 'pending'
})

// 方法
const getStatusBadge = (status) => {
  const badges = {
    pending: 'default',
    running: 'processing',
    completed: 'success',
    failed: 'error',
    cancelled: 'warning'
  }
  return badges[status] || 'default'
}

const getStatusText = (status) => {
  const texts = {
    pending: '等待中',
    running: '运行中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消'
  }
  return texts[status] || status
}

const getProgressStatus = (status) => {
  if (status === 'failed') return 'exception'
  if (status === 'completed') return 'success'
  return 'normal'
}

const formatTime = (timeString) => {
  if (!timeString) return '未知'
  return dayjs(timeString).format('MM-DD HH:mm:ss')
}

const showTaskDetails = (task) => {
  currentTask.value = task
  showModal.value = true
  showDrawer.value = false
}

const showTaskList = () => {
  showDrawer.value = true
}

const handleCancel = () => {
  if (!isRunning.value) {
    showModal.value = false
    currentTask.value = null
  }
}

const handleClose = () => {
  showModal.value = false
  currentTask.value = null
}

const handleMinimize = () => {
  showModal.value = false
  // 不清除currentTask，保持追踪
}

const handleDelete = async () => {
  if (!currentTask.value) return
  
  try {
    await taskStore.deleteTask(currentTask.value.id)
    showModal.value = false
    currentTask.value = null
  } catch (error) {
    console.error('删除任务失败:', error)
  }
}

const refreshTasks = async () => {
  try {
    await taskStore.loadTasks()
  } catch (error) {
    console.error('刷新任务失败:', error)
  }
}

const clearCompleted = () => {
  taskStore.clearCompletedTasks()
}

const startTracking = (taskId) => {
  if (!taskId) return
  
  taskStore.startTaskTracking(
    taskId,
    (task) => {
      // 更新回调
      if (currentTask.value && currentTask.value.id === taskId) {
        currentTask.value = task
      }
      emit('taskUpdate', task)
    },
    (task) => {
      // 完成回调
      if (currentTask.value && currentTask.value.id === taskId) {
        currentTask.value = task
      }
      emit('taskComplete', task)
    },
    (error) => {
      // 错误回调
      emit('taskError', error)
    }
  )
}

// 暴露方法给父组件
const showProgress = (taskId) => {
  if (taskId) {
    const task = tasks.value.find(t => t.id === taskId)
    if (task) {
      currentTask.value = task
      showModal.value = true
      startTracking(taskId)
    }
  }
}

// 监听props.taskId变化
watch(() => props.taskId, (newTaskId) => {
  if (newTaskId && props.autoShow) {
    showProgress(newTaskId)
  }
}, { immediate: true })

// 生命周期
onMounted(() => {
  // 加载任务列表
  taskStore.loadTasks()
})

onUnmounted(() => {
  // 停止所有追踪
  taskStore.stopAllTracking()
})

// 暴露方法
defineExpose({
  showProgress,
  showTaskList
})
</script>

<style scoped>
.floating-tasks {
  position: fixed;
  bottom: 20px;
  right: 20px;
  z-index: 1000;
  cursor: pointer;
}

.floating-icon {
  background: #1890ff;
  color: white;
  padding: 12px 16px;
  border-radius: 20px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  display: flex;
  align-items: center;
  transition: all 0.3s ease;
}

.floating-icon:hover {
  background: #40a9ff;
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
}

.task-list {
  height: 100%;
}

.task-item {
  border: 1px solid #d9d9d9;
  border-radius: 6px;
  padding: 12px;
  margin-bottom: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.task-item:hover {
  border-color: #40a9ff;
  box-shadow: 0 2px 8px rgba(64, 169, 255, 0.2);
}

.task-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.task-name {
  font-weight: 500;
  color: #262626;
}

.task-progress-mini {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
}

.task-progress-mini .ant-progress {
  flex: 1;
  margin-right: 8px;
}

.progress-text {
  font-size: 12px;
  color: #666;
  white-space: nowrap;
}

.task-message {
  font-size: 12px;
  color: #666;
  margin-bottom: 4px;
  line-height: 1.4;
  max-height: 2.8em;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.task-time {
  font-size: 11px;
  color: #999;
  text-align: right;
}

:deep(.ant-progress-inner) {
  background-color: #f5f5f5;
}

:deep(.ant-descriptions-item-label) {
  background-color: #fafafa;
  font-weight: 500;
}
</style>
