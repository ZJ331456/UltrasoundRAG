// API服务层
import axios from 'axios'
import { message } from 'ant-design-vue'

// 创建axios实例 - 统一使用/api/v1/rag前缀
const api = axios.create({
  baseURL: '/api/v1/rag',
  timeout: 120000, // 增加到2分钟
  headers: {
    'Content-Type': 'application/json',
  }
})

// 为特定操作创建长超时实例
const longTimeoutApi = axios.create({
  baseURL: '/api/v1/rag',
  timeout: 3600000, // 1小时超时，用于索引构建等长时间操作
  headers: {
    'Content-Type': 'application/json',
  }
})

// 为两个axios实例添加相同的拦截器
const setupInterceptors = (axiosInstance) => {
  // 请求拦截器
  axiosInstance.interceptors.request.use(
    (config) => {
      // 添加请求时间戳
      config.metadata = { startTime: Date.now() }
      // 调试：打印完整URL
      console.log('API Request:', config.method?.toUpperCase(), config.baseURL + config.url)
      return config
    },
    (error) => {
      return Promise.reject(error)
    }
  )

  // 响应拦截器
  axiosInstance.interceptors.response.use(
    (response) => {
      // 计算请求耗时
      const duration = Date.now() - response.config.metadata.startTime
      console.log(`API请求耗时: ${duration}ms - ${response.config.url}`)
      
      return response
    },
    (error) => {
      // 统一错误处理
      let errorMessage = '请求失败'
      
      if (error.code === 'ECONNABORTED') {
        errorMessage = '请求超时，操作可能需要更长时间'
      } else if (error.response) {
        // 服务器返回错误状态码
        const { status, data } = error.response
        
        switch (status) {
          case 400:
            errorMessage = data.detail || '请求参数错误'
            break
          case 401:
            errorMessage = '未授权，请重新登录'
            break
          case 403:
            errorMessage = '禁止访问'
            break
          case 404:
            errorMessage = '接口不存在'
            break
          case 500:
            errorMessage = '服务器内部错误'
            break
          case 503:
            errorMessage = '服务不可用'
            break
          default:
            errorMessage = data.detail || data.message || `请求失败 (${status})`
        }
      } else if (error.request) {
        // 网络错误
        errorMessage = '网络连接失败，请检查网络或后端服务'
      } else {
        errorMessage = error.message || '未知错误'
      }
      
      // 显示错误消息
      message.error(errorMessage)
      
      return Promise.reject(error)
    }
  )
}

// 为两个实例设置拦截器
setupInterceptors(api)
setupInterceptors(longTimeoutApi)

// API方法
export const searchAPI = {
  // 统一搜索
  search: (params) => api.post('/search', params),
  
  // 基于集合的搜索（新版本推荐）
  collectionSearch: (params) => api.post('/search/collections', params),
  
  // 基于集合的文件上传搜索
  collectionUploadSearch: (formData) => api.post('/search/collections/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
  
  // 获取集合信息（包括类型）
  getCollectionsInfo: () => api.get('/search/collections/info'),
  
  // 多数据库搜索（向后兼容）
  multiSearch: (params) => api.post('/search/multi-database', params),
  
  // 文件上传搜索
  uploadSearch: (formData) => api.post('/search/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
  
  // 多集合文件上传搜索（向后兼容）
  uploadMultiSearch: (formData) => api.post('/search/upload-multi', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
}

export const databaseAPI = {
  // 数据库管理
  list: () => api.get('/databases'),
  create: (data) => api.post('/databases', data),
  update: (name, data) => api.put(`/databases/${name}`, data),
  delete: (name) => api.delete(`/databases/${name}`)
  
  // 注意：文档管理统一使用collectionAPI，避免重复
}

export const collectionAPI = {
  // 集合管理
  list: () => api.get('/collections'),
  create: (data) => api.post('/collections', data, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
  get: (name) => api.get(`/collections/${name}`),
  delete: (name) => api.delete(`/collections/${name}`),
  rebuild: (name) => api.post(`/indexes/rebuild/${name}`), // 更新路径到indexes模块
  
  // 集合源文件统计
  getSourceFilesSummary: () => api.get('/collections/source-files-summary'),
  
  // 集合中的文档管理
  listDocuments: (collectionName, offset = 0, limit = 20) => 
    api.get(`/collections/${collectionName}/documents`, { 
      params: { offset, limit } 
    }),
  addDocument: (collectionName, formData) => api.post(
    `/documents/${collectionName}`, 
    formData,
    { headers: { 'Content-Type': 'multipart/form-data' } }
  ),
  batchAddDocuments: (collectionName, formData) => api.post(
    `/documents/${collectionName}/batch`, 
    formData,
    { headers: { 'Content-Type': 'multipart/form-data' } }
  ),
  updateDocument: (collectionName, documentName, formData) => api.put(
    `/documents/${collectionName}/${documentName}`,
    formData,
    { headers: { 'Content-Type': 'multipart/form-data' } }
  ),
  deleteDocument: (collectionName, documentKey) => 
    api.delete(`/documents/${collectionName}/${documentKey}`),
  
  // 文件管理 - 基于file字段
  getFileList: (collectionName) => 
    api.get(`/collections/${collectionName}/files`),
  getFileStatistics: (collectionName) => 
    api.get(`/collections/${collectionName}/files/statistics`),
  deleteFile: (collectionName, fileValue) => 
    api.delete(`/collections/${collectionName}/files?file_value=${encodeURIComponent(fileValue)}`),
  batchDeleteFiles: (collectionName, fileValues) => 
    api.post(`/collections/${collectionName}/files/batch-delete`, fileValues),
  
  // 高效集合信息获取 - 新增
  getCollectionInfoFast: (collectionName) => 
    api.get(`/collections/${collectionName}/info/fast`),
  getCollectionStatsOptimized: (collectionName) => 
    api.get(`/collections/${collectionName}/stats/optimized`)
}

export const taskAPI = {
  // 任务管理
  list: () => api.get('/tasks'),
  get: (taskId) => api.get(`/tasks/${taskId}`),
  delete: (taskId) => api.delete(`/tasks/${taskId}`)
}

export const systemAPI = {
  // 系统状态
  health: () => api.get('/health/'),
  status: () => api.get('/health/system/status'),
  reloadConfig: () => api.post('/admin/system/reload-config'),
  
  // 索引管理 - 现在返回任务ID，立即响应
  buildIndexes: (params) => api.post('/indexes/build', params),
  updateDocuments: (docs, incremental = true) => 
    api.post('/indexes/documents/update', docs, { params: { incremental } }),
  
  // 测试 - 使用长超时实例
  runTests: (params) => longTimeoutApi.post('/admin/tests/run', params),
  
  // 管理功能
  clearCache: () => api.post('/admin/cache/clear'),
  reloadModels: () => api.post('/admin/models/reload'),
  getAlerts: () => api.get('/admin/alerts')
}

export const fileAPI = {
  // 增强的文件管理
  uploadImageFolder: (formData) => api.post('/files/upload-image-folder', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
  getImageFile: (imageHash, size = 'original') => api.get(`/files/image/${imageHash}`, {
    params: { size }
  }),
  getStorageInfo: () => api.get('/files/storage-info')
}

export { api }
export default api
