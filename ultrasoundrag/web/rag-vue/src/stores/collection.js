// 集合管理状态
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { collectionAPI } from '../api/index.js'
import { message } from 'ant-design-vue'

export const useCollectionStore = defineStore('collection', () => {
  // 状态
  const collections = ref([])
  const currentCollection = ref(null)
  const loading = ref(false)
  const error = ref(null)

  // 计算属性
  const collectionCount = computed(() => collections.value.length)
  const hasCollections = computed(() => collections.value.length > 0)

  // 方法
  const loadCollections = async () => {
    try {
      loading.value = true
      error.value = null
      
      const response = await collectionAPI.list()
      
      if (response.data.success) {
        collections.value = response.data.data?.collections || []
      } else {
        throw new Error(response.data.error || '加载集合列表失败')
      }
      
      return response.data
    } catch (err) {
      error.value = err.message || '加载集合列表失败'
      message.error(error.value)
      throw err
    } finally {
      loading.value = false
    }
  }

  const createCollection = async (collectionData) => {
    try {
      loading.value = true
      
      // 创建FormData
      const formData = new FormData()
      formData.append('name', collectionData.name)
      formData.append('type', collectionData.type)
      formData.append('description', collectionData.description || '')
      
      const response = await collectionAPI.create(formData)
      
      if (response.data.success) {
        message.success(response.data.message || '集合创建成功')
        await loadCollections() // 重新加载列表
        return response.data
      } else {
        throw new Error(response.data.error || '创建集合失败')
      }
    } catch (err) {
      error.value = err.message || '创建集合失败'
      message.error(error.value)
      throw err
    } finally {
      loading.value = false
    }
  }

  const getCollectionInfo = async (collectionName) => {
    try {
      loading.value = true
      
      const response = await collectionAPI.get(collectionName)
      
      if (response.data.success) {
        currentCollection.value = response.data.data
        return response.data
      } else {
        throw new Error(response.data.error || '获取集合信息失败')
      }
    } catch (err) {
      error.value = err.message || '获取集合信息失败'
      message.error(error.value)
      throw err
    } finally {
      loading.value = false
    }
  }

  const deleteCollection = async (collectionName) => {
    try {
      loading.value = true
      
      const response = await collectionAPI.delete(collectionName)
      
      if (response.data.success) {
        message.success(response.data.message || '集合删除成功')
        await loadCollections() // 重新加载列表
        return response.data
      } else {
        throw new Error(response.data.error || '删除集合失败')
      }
    } catch (err) {
      error.value = err.message || '删除集合失败'
      message.error(error.value)
      throw err
    } finally {
      loading.value = false
    }
  }

  const rebuildCollection = async (collectionName) => {
    try {
      loading.value = true
      
      const response = await collectionAPI.rebuild(collectionName)
      
      if (response.data.success) {
        const taskId = response.data.task_id
        message.success(response.data.message || '集合重建任务已启动')
        
        return {
          ...response.data,
          taskId: taskId
        }
      } else {
        throw new Error(response.data.message || '启动集合重建任务失败')
      }
    } catch (err) {
      error.value = err.message || '启动集合重建失败'
      message.error(error.value)
      throw err
    } finally {
      loading.value = false
    }
  }

  const addDocumentToCollection = async (collectionName, file, docType = 'md') => {
    try {
      loading.value = true
      
      const formData = new FormData()
      formData.append('file', file)
      formData.append('doc_type', docType)
      
      const response = await collectionAPI.addDocument(collectionName, formData)
      
      if (response.data.success) {
        message.success(response.data.message || '文档添加成功')
        return response.data
      } else {
        throw new Error(response.data.error || '添加文档失败')
      }
    } catch (err) {
      error.value = err.message || '添加文档失败'
      message.error(error.value)
      throw err
    } finally {
      loading.value = false
    }
  }

  const updateDocumentInCollection = async (collectionName, documentName, file, docType = 'md') => {
    try {
      loading.value = true
      
      const formData = new FormData()
      formData.append('file', file)
      formData.append('doc_type', docType)
      
      const response = await collectionAPI.updateDocument(collectionName, documentName, formData)
      
      if (response.data.success) {
        message.success(response.data.message || '文档更新成功')
        return response.data
      } else {
        throw new Error(response.data.error || '更新文档失败')
      }
    } catch (err) {
      error.value = err.message || '更新文档失败'
      message.error(error.value)
      throw err
    } finally {
      loading.value = false
    }
  }

  const deleteDocumentFromCollection = async (collectionName, documentName) => {
    try {
      loading.value = true
      
      const response = await collectionAPI.deleteDocument(collectionName, documentName)
      
      if (response.data.success) {
        message.success(response.data.message || '文档删除成功')
        return response.data
      } else {
        throw new Error(response.data.error || '删除文档失败')
      }
    } catch (err) {
      error.value = err.message || '删除文档失败'
      message.error(error.value)
      throw err
    } finally {
      loading.value = false
    }
  }

  // 文件管理方法 - 基于file字段
  const getCollectionFileList = async (collectionName) => {
    try {
      loading.value = true
      
      const response = await collectionAPI.getFileList(collectionName)
      
      if (response.data.success) {
        return response.data
      } else {
        throw new Error(response.data.error || '获取文件列表失败')
      }
    } catch (err) {
      error.value = err.message || '获取文件列表失败'
      message.error(error.value)
      throw err
    } finally {
      loading.value = false
    }
  }

  const getCollectionFileStatistics = async (collectionName) => {
    try {
      loading.value = true
      
      const response = await collectionAPI.getFileStatistics(collectionName)
      
      if (response.data.success) {
        return response.data
      } else {
        throw new Error(response.data.error || '获取文件统计信息失败')
      }
    } catch (err) {
      error.value = err.message || '获取文件统计信息失败'
      message.error(error.value)
      throw err
    } finally {
      loading.value = false
    }
  }

  const getCollectionStatsOptimized = async (collectionName) => {
    try {
      loading.value = true
      
      const response = await collectionAPI.getCollectionStatsOptimized(collectionName)
      
      if (response.data.success) {
        return response.data
      } else {
        throw new Error(response.data.error || '获取集合统计信息失败')
      }
    } catch (err) {
      error.value = err.message || '获取集合统计信息失败'
      message.error(error.value)
      throw err
    } finally {
      loading.value = false
    }
  }

  const deleteFileFromCollection = async (collectionName, fileValue) => {
    try {
      loading.value = true
      
      const response = await collectionAPI.deleteFile(collectionName, fileValue)
      
      if (response.data.success) {
        message.success(response.data.message || '文件删除成功')
        return response.data
      } else {
        throw new Error(response.data.error || '删除文件失败')
      }
    } catch (err) {
      error.value = err.message || '删除文件失败'
      message.error(error.value)
      throw err
    } finally {
      loading.value = false
    }
  }

  const batchDeleteFilesFromCollection = async (collectionName, fileValues) => {
    try {
      loading.value = true
      
      const response = await collectionAPI.batchDeleteFiles(collectionName, fileValues)
      
      if (response.data.success) {
        message.success(response.data.message || '批量删除文件成功')
        return response.data
      } else {
        throw new Error(response.data.error || '批量删除文件失败')
      }
    } catch (err) {
      error.value = err.message || '批量删除文件失败'
      message.error(error.value)
      throw err
    } finally {
      loading.value = false
    }
  }

  // 清除当前集合信息
  const clearCurrentCollection = () => {
    currentCollection.value = null
  }

  // 清除错误
  const clearError = () => {
    error.value = null
  }

  // 返回状态和方法
  return {
    // 状态
    collections,
    currentCollection,
    loading,
    error,
    
    // 计算属性
    collectionCount,
    hasCollections,
    
    // 方法
    loadCollections,
    createCollection,
    getCollectionInfo,
    deleteCollection,
    rebuildCollection,
    addDocumentToCollection,
    updateDocumentInCollection,
    deleteDocumentFromCollection,
    
    // 文件管理方法
    getCollectionFileList,
    getCollectionFileStatistics,
    deleteFileFromCollection,
    batchDeleteFilesFromCollection,
    
    // 统计方法
    getCollectionStatsOptimized,
    
    clearCurrentCollection,
    clearError
  }
})
