<template>
  <div class="data-manager">
    <a-page-header
      title="数据管理"
      sub-title="管理Milvus数据库中的集合和文件"
      style="padding: 0 0 24px 0;"
    >
      <template #extra>
        <a-space>
          <a-button @click="refreshAll" :loading="loading">
            <ReloadOutlined />
            刷新全部
          </a-button>
          <a-button type="primary" @click="showCreateModal = true">
            <PlusOutlined />
            新建
          </a-button>
        </a-space>
      </template>
    </a-page-header>

    <!-- 统计信息 -->
    <a-row :gutter="[16, 16]" style="margin-bottom: 24px;">
      <a-col :xs="24" :sm="6">
        <a-card>
          <a-statistic
            title="Milvus数据库"
            value="ultrasound_vector"
            :value-style="{ color: '#1890ff' }"
          />
        </a-card>
      </a-col>
      <a-col :xs="24" :sm="6">
        <a-card>
          <a-statistic
            title="集合"
            :value="collectionCount"
            :value-style="{ color: '#52c41a' }"
            suffix="个"
          />
        </a-card>
      </a-col>
      <a-col :xs="24" :sm="6">
        <a-card>
          <a-statistic
            title="文件"
            :value="fileCount"
            :value-style="{ color: '#722ed1' }"
            suffix="个"
          />
        </a-card>
      </a-col>
      <a-col :xs="24" :sm="6">
        <a-card>
          <a-statistic
            title="总记录"
            :value="totalRecords"
            :value-style="{ color: '#eb2f96' }"
            suffix="条"
          />
        </a-card>
      </a-col>
    </a-row>

    <!-- 主要内容区域 -->
    <a-row :gutter="[16, 16]">
      <!-- 左侧树形导航 -->
      <a-col :xs="24" :lg="8">
        <a-card title="集合层级" :bordered="false">
          <a-tree
            :tree-data="treeData"
            :selected-keys="selectedKeys"
            :expanded-keys="expandedKeys"
            @select="onTreeSelect"
            @expand="onTreeExpand"
            :load-data="onLoadData"
            show-line
            show-icon
          >
            <template #icon="{ type }">
              <DatabaseOutlined v-if="type === 'database'" />
              <FolderOutlined v-else />
            </template>
            <template #title="{ title, type, data }">
              <div class="tree-node">
                <span class="node-title">{{ title }}</span>
                <a-space class="node-actions">
                  <a-button 
                    size="small" 
                    type="text"
                    @click.stop="handleNodeAction('view', data)"
                    title="查看详情"
                  >
                    <EyeOutlined />
                  </a-button>
                  <a-button 
                    size="small" 
                    type="text"
                    @click.stop="handleNodeAction('rebuild', data)"
                    :loading="rebuilding[data?.name]"
                    title="重建索引"
                    v-if="type === 'collection'"
                  >
                    <ReloadOutlined />
                  </a-button>
                  <a-button 
                    size="small" 
                    type="text"
                    danger
                    @click.stop="handleNodeAction('delete', data)"
                    title="删除"
                  >
                    <DeleteOutlined />
                  </a-button>
                </a-space>
              </div>
            </template>
          </a-tree>
        </a-card>
      </a-col>

      <!-- 右侧详情区域 -->
      <a-col :xs="24" :lg="16">
        <a-card :title="detailTitle" :bordered="false">
          <!-- 数据库详情 -->
          <div v-if="selectedType === 'database' && selectedDatabase">
            <a-descriptions :column="2" bordered size="small">
              <a-descriptions-item label="数据库名称">
                {{ selectedDatabase.name }}
              </a-descriptions-item>
              <a-descriptions-item label="状态">
                <a-badge 
                  :status="selectedDatabase.enabled ? 'success' : 'error'"
                  :text="selectedDatabase.enabled ? '启用' : '禁用'"
                />
              </a-descriptions-item>
              <a-descriptions-item label="集合数量">
                {{ selectedDatabase.collections?.length || 0 }}
              </a-descriptions-item>
              <a-descriptions-item label="创建时间">
                {{ formatDate(selectedDatabase.created_at) }}
              </a-descriptions-item>
              <a-descriptions-item label="描述" :span="2">
                {{ selectedDatabase.description || '无描述' }}
              </a-descriptions-item>
            </a-descriptions>
            
            <!-- 数据库中的集合列表 -->
            <a-divider>
              <span>集合列表</span>
            </a-divider>
            <a-table
              :columns="collectionColumns"
              :data-source="selectedDatabase.collections || []"
              :pagination="false"
              size="small"
              row-key="name"
            >
              <template #bodyCell="{ column, record }">
                <template v-if="column.key === 'type'">
                  <a-tag :color="getTypeColor(record.type)">
                    {{ getTypeName(record.type) }}
                  </a-tag>
                </template>
                <template v-if="column.key === 'document_count'">
                  <a-tag color="green">{{ record.document_count || 0 }}</a-tag>
                </template>
                <template v-if="column.key === 'actions'">
                  <a-space>
                    <a-button 
                      size="small" 
                      @click="selectCollection(record)"
                    >
                      查看详情
                    </a-button>
                    <a-button 
                      size="small" 
                      danger
                      @click="confirmDelete({ type: 'collection', name: record.name })"
                    >
                      删除
                    </a-button>
                  </a-space>
                </template>
              </template>
            </a-table>
          </div>

          <!-- 集合详情 -->
          <div v-if="selectedType === 'collection' && selectedCollection">
            <a-descriptions :column="2" bordered size="small">
              <a-descriptions-item label="集合名称">
                {{ selectedCollection.name }}
              </a-descriptions-item>
              <a-descriptions-item label="类型">
                <a-tag :color="getTypeColor(selectedCollection.type)">
                  {{ getTypeName(selectedCollection.type) }}
                </a-tag>
              </a-descriptions-item>
              <a-descriptions-item label="集合数量">
                {{ selectedCollection.document_count || 0 }}
              </a-descriptions-item>
              <a-descriptions-item label="源文件数量">
                {{ selectedCollection.file_count || 0 }}
              </a-descriptions-item>
              <a-descriptions-item label="状态">
                <a-badge 
                  :status="selectedCollection.status === 'active' ? 'success' : 'default'"
                  :text="selectedCollection.status === 'active' ? '活跃' : '未知'"
                />
              </a-descriptions-item>
            </a-descriptions>

            <!-- 源文件列表 -->
            <a-divider>
              <span>源文件列表 (用于删除操作)</span>
              <a-space style="margin-left: 16px;">
                <a-button 
                  type="primary" 
                  size="small"
                  @click="showAddDocumentModal = true"
                >
                  <PlusOutlined />
                  添加文档
                </a-button>
                <a-button 
                  size="small"
                  @click="loadCollectionFiles(selectedCollection.name)"
                >
                  <ReloadOutlined />
                  刷新
                </a-button>
              </a-space>
            </a-divider>
            <a-table
              :columns="fileColumns"
              :data-source="collectionFiles"
              :loading="loadingFiles"
              :pagination="filePagination"
              size="small"
              row-key="file"
            >
              <template #bodyCell="{ column, record }">
                <template v-if="column.key === 'file'">
                  <a-tag color="blue">{{ record.file }}</a-tag>
                </template>
                <template v-if="column.key === 'count'">
                  <a-tag color="green">{{ record.count || 0 }}</a-tag>
                </template>
                <template v-if="column.key === 'actions'">
                  <a-space>
                    <a-button 
                      size="small" 
                      @click="viewFileDetails(record)"
                    >
                      查看
                    </a-button>
                    <a-button 
                      size="small" 
                      @click="editFile(record)"
                    >
                      编辑
                    </a-button>
                    <a-button 
                      size="small" 
                      danger
                      @click="deleteFile(record)"
                    >
                      删除
                    </a-button>
                  </a-space>
                </template>
              </template>
            </a-table>
          </div>

          <!-- 文件详情 -->
          <div v-if="selectedType === 'file' && selectedFile">
            <a-descriptions :column="2" bordered size="small">
              <a-descriptions-item label="文件名">
                <a-tag color="blue">{{ selectedFile.file }}</a-tag>
              </a-descriptions-item>
              <a-descriptions-item label="记录数量">
                <a-badge :count="selectedFile.count" :number-style="{ backgroundColor: '#52c41a' }" />
              </a-descriptions-item>
              <a-descriptions-item label="所属集合">
                {{ selectedFile.collection }}
              </a-descriptions-item>
              <a-descriptions-item label="集合类型">
                <a-tag :color="getTypeColor(selectedFile.collectionType)">
                  {{ getTypeName(selectedFile.collectionType) }}
                </a-tag>
              </a-descriptions-item>
            </a-descriptions>
          </div>

          <!-- 空状态 -->
          <a-empty v-if="!selectedType" description="请从左侧选择要查看的项目" />
        </a-card>
      </a-col>
    </a-row>

    <!-- 创建模态框 -->
    <a-modal
      v-model:open="showCreateModal"
      title="新建项目"
      @ok="handleCreate"
      :confirm-loading="creating"
    >
      <a-form :model="newItem" layout="vertical">
        <a-form-item label="类型" required>
          <a-select v-model:value="newItem.type" @change="onCreateTypeChange">
            <a-select-option value="database">数据库</a-select-option>
            <a-select-option value="collection">集合</a-select-option>
          </a-select>
        </a-form-item>
        
        <a-form-item label="名称" required>
          <a-input 
            v-model:value="newItem.name" 
            :placeholder="getCreatePlaceholder()"
          />
        </a-form-item>
        
        <a-form-item label="描述">
          <a-textarea 
            v-model:value="newItem.description" 
            :rows="3"
          />
        </a-form-item>
        
        <a-form-item v-if="newItem.type === 'collection'" label="集合类型" required>
          <a-select v-model:value="newItem.collectionType">
            <a-select-option value="md">Markdown文档</a-select-option>
            <a-select-option value="image">图片文件</a-select-option>
            <a-select-option value="pdf">PDF文档</a-select-option>
          </a-select>
        </a-form-item>
      </a-form>
    </a-modal>

    <!-- 添加文档模态框 -->
    <a-modal
      v-model:open="showAddDocumentModal"
      title="添加文档"
      @ok="handleAddDocument"
      :confirm-loading="uploadingDocument"
    >
      <a-form layout="vertical">
        <a-form-item label="集合名称">
          <a-input :value="selectedCollection?.name" disabled />
        </a-form-item>
        
        <a-form-item label="文档类型" required>
          <a-select v-model:value="documentFormData.docType">
            <a-select-option value="md">Markdown文档</a-select-option>
            <a-select-option value="pdf">PDF文档</a-select-option>
            <a-select-option value="image">图片文件</a-select-option>
          </a-select>
        </a-form-item>
        
        <a-form-item label="选择文件" required>
          <a-upload
            :file-list="[]"
            :before-upload="(file) => { documentFormData.file = file; return false }"
            :max-count="1"
          >
            <a-button>
              <UploadOutlined />
              选择文件
            </a-button>
          </a-upload>
          <div v-if="documentFormData.file" style="margin-top: 8px;">
            已选择：{{ documentFormData.file.name }}
          </div>
        </a-form-item>
      </a-form>
    </a-modal>

    <!-- 编辑文件模态框 -->
    <a-modal
      v-model:open="showEditFileModal"
      title="更新文件"
      @ok="handleEditFile"
      :confirm-loading="uploadingDocument"
    >
      <a-form layout="vertical">
        <a-form-item label="当前文件">
          <a-input :value="currentEditFile?.file" disabled />
        </a-form-item>
        
        <a-form-item label="集合名称">
          <a-input :value="currentEditFile?.collection" disabled />
        </a-form-item>
        
        <a-form-item label="文档类型" required>
          <a-select v-model:value="documentFormData.docType">
            <a-select-option value="md">Markdown文档</a-select-option>
            <a-select-option value="pdf">PDF文档</a-select-option>
            <a-select-option value="image">图片文件</a-select-option>
          </a-select>
        </a-form-item>
        
        <a-form-item label="选择新文件" required>
          <a-upload
            :file-list="[]"
            :before-upload="(file) => { documentFormData.file = file; return false }"
            :max-count="1"
          >
            <a-button>
              <UploadOutlined />
              选择文件
            </a-button>
          </a-upload>
          <div v-if="documentFormData.file" style="margin-top: 8px;">
            已选择：{{ documentFormData.file.name }}
          </div>
        </a-form-item>
      </a-form>
    </a-modal>

    <!-- 任务进度追踪组件 -->
    <TaskProgress ref="taskProgressRef" @task-complete="handleTaskComplete" />
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { message, Modal } from 'ant-design-vue'
import {
  PlusOutlined,
  ReloadOutlined,
  EyeOutlined,
  DeleteOutlined,
  DatabaseOutlined,
  FolderOutlined,
  FileOutlined,
  UploadOutlined
} from '@ant-design/icons-vue'
import { useSystemStore } from '../stores/system'
import { useCollectionStore } from '../stores/collection'
import { systemAPI, collectionAPI, databaseAPI } from '../api/index.js'
import TaskProgress from '../components/TaskProgress.vue'
import dayjs from 'dayjs'

// 状态管理
const systemStore = useSystemStore()
const collectionStore = useCollectionStore()

// 响应式数据
const loading = ref(false)
const creating = ref(false)
const loadingFiles = ref(false)
const rebuilding = reactive({})
const taskProgressRef = ref(null)

// 树形数据
const treeData = ref([])
const selectedKeys = ref([])
const expandedKeys = ref([])

// 详情数据
const selectedType = ref('')
const selectedDatabase = ref(null)
const selectedCollection = ref(null)
const selectedFile = ref(null)
const collectionFiles = ref([])

// 创建相关
const showCreateModal = ref(false)
const newItem = ref({
  type: '',
  name: '',
  description: '',
  collectionType: ''
})

// 文档管理相关
const showAddDocumentModal = ref(false)
const showEditFileModal = ref(false)
const currentEditFile = ref(null)
const uploadingDocument = ref(false)
const documentFormData = ref({
  file: null,
  docType: 'md'
})

// 分页
const filePagination = reactive({
  current: 1,
  pageSize: 10,
  total: 0
})

// 计算属性
const databaseCount = computed(() => systemStore.databases.length)
const collectionCount = computed(() => collectionStore.collectionCount)
const fileCount = computed(() => {
  // 如果有选中的集合，显示该集合的文件数量
  if (selectedType.value === 'collection' && collectionFiles.value.length > 0) {
    return collectionFiles.value.length
  }
  // 否则显示总的文件数量估算
  return collectionStore.collections.reduce((total, collection) => {
    // 估算：假设每个集合平均有 document_count / 20 个文件
    const estimatedFiles = Math.max(1, Math.floor((collection.document_count || 0) / 20))
    return total + estimatedFiles
  }, 0)
})
const totalRecords = computed(() => {
  return collectionStore.collections.reduce((total, collection) => {
    return total + (collection.document_count || 0)
  }, 0)
})

const detailTitle = computed(() => {
  if (selectedType.value === 'database') return '数据库详情'
  if (selectedType.value === 'collection') return '集合详情'
  if (selectedType.value === 'file') return '文件详情'
  return '数据管理'
})

// 文件表格列
const fileColumns = [
  { title: '源文件名', dataIndex: 'file', key: 'file' },
  { title: '记录数', dataIndex: 'count', key: 'count', width: 100 },
  { title: '操作', key: 'actions', width: 120 }
]

// 集合表格列
const collectionColumns = [
  { title: '集合名称', dataIndex: 'name', key: 'name' },
  { title: '类型', dataIndex: 'type', key: 'type', width: 100 },
  { title: '文档数量', dataIndex: 'document_count', key: 'document_count', width: 120 },
  { title: '操作', key: 'actions', width: 150 }
]

// 方法
const refreshAll = async () => {
  try {
    loading.value = true
    await Promise.all([
      systemStore.loadDatabases(),
      collectionStore.loadCollections()
    ])
    buildTreeData()
    
    // 如果当前有选中的集合，刷新其详细信息
    if (selectedCollection.value && selectedCollection.value.name) {
      await Promise.all([
        loadCollectionFiles(selectedCollection.value.name),
        loadCollectionInfoFast(selectedCollection.value.name)
      ])
    }
  } catch (error) {
    console.error('刷新数据失败:', error)
  } finally {
    loading.value = false
  }
}

const buildTreeData = () => {
  const databases = systemStore.databases || []
  const collections = collectionStore.collections || []
  
  console.log('构建树形数据 - 数据库:', databases)
  console.log('构建树形数据 - 集合:', collections)
  
  treeData.value = databases.map(db => {
    // 计算属于该数据库的集合
    const dbCollections = collections.filter(col => {
      // 修复过滤逻辑，因为集合可能没有database字段
      return col.database === db.name || 
             col.name?.includes(db.name) || 
             (databases.length === 1) // 如果只有一个数据库，包含所有集合
    })
    
    return {
      title: db.name || '未知数据库',
      key: `db-${db.name || 'unknown'}`,
      type: 'database',
      data: {
        ...db,
        collections: dbCollections // 添加collections字段
      },
      children: dbCollections.map(col => {
        console.log('处理集合数据:', col)
        return {
          title: `${col.name || '未知集合'} (${col.document_count > 0 ? col.document_count : col.document_count === -1 ? '未知' : '0'}条文档)`,
          key: `col-${col.name || 'unknown'}`,
          type: 'collection',
          data: col,
          isLeaf: false
        }
      })
    }
  })
  
  // 如果没有数据库但有集合，创建默认分组
  if (databases.length === 0 && collections.length > 0) {
    treeData.value = [{
      title: '默认数据库',
      key: 'db-default',
      type: 'database',
      data: { 
        name: 'default', 
        description: '默认数据库',
        collections: collections // 添加collections字段
      },
      children: collections.map(col => {
        console.log('默认数据库 - 处理集合数据:', col)
        return {
          title: `${col.name || '未知集合'} (${col.document_count > 0 ? col.document_count : col.document_count === -1 ? '未知' : '0'}条文档)`,
          key: `col-${col.name || 'unknown'}`,
          type: 'collection', 
          data: col,
          isLeaf: false
        }
      })
    }]
  }
}

const onTreeSelect = (selectedKeys, { node }) => {
  selectedKeys.value = selectedKeys
  const nodeData = node.dataRef
  
  if (nodeData.type === 'database') {
    selectedType.value = 'database'
    selectedDatabase.value = nodeData.data
    selectedCollection.value = null
    selectedFile.value = null
  } else if (nodeData.type === 'collection') {
    selectedType.value = 'collection'
    selectedCollection.value = nodeData.data
    selectedDatabase.value = null
    selectedFile.value = null
    loadCollectionFiles(nodeData.data.name)
    loadCollectionInfoFast(nodeData.data.name) // 加载快速集合信息
  }
}

const onTreeExpand = (expandedKeys) => {
  expandedKeys.value = expandedKeys
}

const onLoadData = async (treeNode) => {
  if (treeNode.dataRef.type === 'collection') {
    // 加载集合的文件列表
    await loadCollectionFiles(treeNode.dataRef.data.name)
    
    // 将文件添加到树中
    const files = collectionFiles.value.map(file => ({
      title: file.file,
      key: `file-${file.file}`,
      type: 'file',
      data: { ...file, collection: treeNode.dataRef.data.name },
      icon: FileOutlined,
      isLeaf: true
    }))
    
    treeNode.dataRef.children = files
    return Promise.resolve()
  }
  return Promise.resolve()
}

const loadCollectionFiles = async (collectionName) => {
  try {
    loadingFiles.value = true
    // 使用优化的API方法，减少查询次数
    const [listResponse, statsResponse] = await Promise.all([
      collectionStore.getCollectionFileList(collectionName),
      collectionStore.getCollectionFileStatistics(collectionName) // 已优化，小集合精确统计，大集合采样估算
    ])

    if (listResponse.success && statsResponse.success) {
      const files = listResponse.data.file_list || []
      const stats = statsResponse.data
      
      console.log('文件列表响应:', listResponse.data)
      console.log('文件统计响应:', stats)
      
      collectionFiles.value = files.map(file => ({
        file,
        count: stats.file_counts?.[file] || 0,
        collection: collectionName,
        collectionType: stats.collection_type,
        totalRecords: stats.total_records
      }))
      
      filePagination.total = collectionFiles.value.length
      console.log(`集合 ${collectionName} 的文件列表:`, collectionFiles.value)
    } else {
      console.error('获取文件列表失败:', listResponse?.error || statsResponse?.error)
      console.error('listResponse:', listResponse)
      console.error('statsResponse:', statsResponse)
      collectionFiles.value = []
    }
  } catch (error) {
    console.error('加载文件列表失败:', error)
    collectionFiles.value = []
  } finally {
    loadingFiles.value = false
  }
}

const loadCollectionInfoFast = async (collectionName) => {
  try {
    console.log('开始加载集合信息:', collectionName)
    // 获取集合信息、统计信息和文件统计信息
    const [infoResponse, statsResponse, fileStatsResponse] = await Promise.all([
      collectionAPI.get(collectionName).catch(() => null),
      collectionStore.getCollectionStatsOptimized ? 
        collectionStore.getCollectionStatsOptimized(collectionName).catch(() => null) : 
        Promise.resolve(null),
      collectionStore.getCollectionFileStatistics ? 
        collectionStore.getCollectionFileStatistics(collectionName).catch(() => null) : 
        Promise.resolve(null)
    ])
    
    console.log('API响应:', { infoResponse, statsResponse, fileStatsResponse })
    
    let collectionInfo = {}
    
    // 处理基本信息
    if (infoResponse?.success) {
      collectionInfo = { ...infoResponse.data }
    }
    
    // 处理统计信息
    if (statsResponse?.success) {
      const stats = statsResponse.data
      collectionInfo.document_count = stats.document_count || stats.total_records || 0
      collectionInfo.file_count = stats.file_count || 0
    }
    
    // 处理文件统计信息
    if (fileStatsResponse?.success) {
      const fileStats = fileStatsResponse.data
      collectionInfo.file_count = fileStats.total_files || 0
      collectionInfo.file_list = fileStats.file_list || []
      console.log('文件统计信息:', fileStats)
    } else {
      console.warn('文件统计信息获取失败:', fileStatsResponse)
    }
    
    // 如果没有统计信息，尝试快速信息接口
    if (!collectionInfo.document_count && collectionAPI.getCollectionInfoFast) {
      try {
        const fastResponse = await collectionAPI.getCollectionInfoFast(collectionName)
        if (fastResponse.data.success) {
          const fastInfo = fastResponse.data.data
          collectionInfo.document_count = fastInfo.document_count || fastInfo.total_records || 0
        }
      } catch (error) {
        console.warn('快速信息接口失败:', error)
      }
    }
    
    // 更新选中的集合信息
    if (selectedCollection.value && selectedCollection.value.name === collectionName) {
      // 使用Object.assign确保响应式更新
      Object.assign(selectedCollection.value, {
        ...collectionInfo,
        // 保留原有字段
        name: collectionName,
        database: selectedCollection.value.database,
        // 确保document_count不为undefined
        document_count: collectionInfo.document_count || 0,
        // 确保file_count不为undefined
        file_count: collectionInfo.file_count || 0
      })
      console.log('更新后的集合信息:', selectedCollection.value)
    }
  } catch (error) {
    console.error('加载集合快速信息失败:', error)
  }
}

const handleNodeAction = (action, data) => {
  if (action === 'view') {
    // 查看详情逻辑已在onTreeSelect中处理
  } else if (action === 'rebuild') {
    // 修复名称获取逻辑
    let name = data.name
    if (!name && data.collection_name) {
      name = data.collection_name
    }
    if (!name && data.title) {
      name = data.title.split(' (')[0]
    }
    rebuildCollection(name)
  } else if (action === 'delete') {
    confirmDelete(data)
  }
}

const rebuildCollection = async (collectionName) => {
  try {
    rebuilding[collectionName] = true
    const result = await collectionStore.rebuildCollection(collectionName)
    
    if (result.taskId && taskProgressRef.value) {
      taskProgressRef.value.showProgress(result.taskId)
    }
  } catch (error) {
    console.error('重建集合失败:', error)
  } finally {
    rebuilding[collectionName] = false
  }
}

const confirmDelete = (data) => {
  const type = data.type || 'item'
  // 修复名称获取逻辑
  let name = data.name || data.file
  if (!name && data.collection_name) {
    name = data.collection_name
  }
  if (!name && data.title) {
    // 从title中提取名称（去掉文档数量部分）
    name = data.title.split(' (')[0]
  }
  
  console.log('删除确认 - 原始数据:', data)
  console.log('删除确认 - 提取的名称:', name)
  
  Modal.confirm({
    title: '确认删除',
    content: `确定要删除${type === 'file' ? '文件' : type === 'collection' ? '集合' : '数据库'} "${name || '未知项目'}" 吗？`,
    okText: '确认删除',
    okType: 'danger',
    onOk: () => deleteItem(data)
  })
}

const deleteItem = async (data) => {
  try {
    // 修复名称获取逻辑，与confirmDelete保持一致
    let name = data.name || data.file
    if (!name && data.collection_name) {
      name = data.collection_name
    }
    if (!name && data.title) {
      name = data.title.split(' (')[0]
    }
    
    console.log('删除操作 - 原始数据:', data)
    console.log('删除操作 - 提取的名称:', name)
    
    if (data.type === 'file') {
      await collectionStore.deleteFileFromCollection(data.collection, data.file)
      // 删除文件后，刷新当前集合的详细信息
      if (selectedCollection.value && selectedCollection.value.name === data.collection) {
        await Promise.all([
          loadCollectionFiles(data.collection),
          loadCollectionInfoFast(data.collection),
          collectionStore.loadCollections() // 刷新集合列表
        ])
        // 重新构建树形数据以更新显示
        buildTreeData()
      }
    } else if (data.type === 'collection') {
      await collectionStore.deleteCollection(name)
    } else if (data.type === 'database') {
      // 删除数据库（实际是删除集合）
      await databaseAPI.delete(name)
    }
    
    message.success('删除成功')
    await refreshAll()
  } catch (error) {
    console.error('删除失败:', error)
    message.error(`删除失败: ${error.message || '未知错误'}`)
  }
}

const selectCollection = (collection) => {
  selectedType.value = 'collection'
  selectedCollection.value = collection
  selectedDatabase.value = null
  selectedFile.value = null
  selectedKeys.value = [`col-${collection.name}`]
  loadCollectionFiles(collection.name)
  loadCollectionInfoFast(collection.name)
}

const viewFileDetails = (file) => {
  selectedType.value = 'file'
  selectedFile.value = file
  selectedKeys.value = [`file-${file.file}`]
}

const deleteFile = (file) => {
  confirmDelete({ type: 'file', file: file.file, collection: file.collection })
}

const editFile = (file) => {
  currentEditFile.value = file
  showEditFileModal.value = true
}

const handleAddDocument = async () => {
  if (!selectedCollection.value || !documentFormData.value.file) {
    message.error('请选择集合和文件')
    return
  }
  
  try {
    uploadingDocument.value = true
    await collectionStore.addDocumentToCollection(
      selectedCollection.value.name, 
      documentFormData.value.file,
      documentFormData.value.docType
    )
    
    showAddDocumentModal.value = false
    documentFormData.value = { file: null, docType: 'md' }
    
    // 添加文档后，刷新集合信息和文件列表
    await Promise.all([
      loadCollectionFiles(selectedCollection.value.name),
      loadCollectionInfoFast(selectedCollection.value.name),
      collectionStore.loadCollections() // 刷新集合列表
    ])
    
    // 重新构建树形数据以更新显示
    buildTreeData()
    
    message.success('文档添加成功')
  } catch (error) {
    console.error('添加文档失败:', error)
    message.error('添加文档失败')
  } finally {
    uploadingDocument.value = false
  }
}

const handleEditFile = async () => {
  if (!currentEditFile.value || !documentFormData.value.file) {
    message.error('请选择要更新的文件')
    return
  }
  
  try {
    uploadingDocument.value = true
    await collectionStore.updateDocumentInCollection(
      currentEditFile.value.collection,
      currentEditFile.value.file,
      documentFormData.value.file,
      documentFormData.value.docType
    )
    
    showEditFileModal.value = false
    currentEditFile.value = null
    documentFormData.value = { file: null, docType: 'md' }
    await loadCollectionFiles(selectedCollection.value.name)
    message.success('文件更新成功')
  } catch (error) {
    console.error('更新文件失败:', error)
    message.error('更新文件失败')
  } finally {
    uploadingDocument.value = false
  }
}

const onCreateTypeChange = () => {
  newItem.value.name = ''
  newItem.value.description = ''
  newItem.value.collectionType = ''
}

const getCreatePlaceholder = () => {
  if (newItem.value.type === 'database') return '输入数据库名称'
  if (newItem.value.type === 'collection') return '输入集合名称'
  return '输入名称'
}

const handleCreate = async () => {
  try {
    creating.value = true
    
    if (newItem.value.type === 'database') {
      // 创建数据库逻辑
    } else if (newItem.value.type === 'collection') {
      await collectionStore.createCollection({
        name: newItem.value.name,
        type: newItem.value.collectionType,
        description: newItem.value.description
      })
    }
    
    message.success('创建成功')
    showCreateModal.value = false
    newItem.value = { type: '', name: '', description: '', collectionType: '' }
    await refreshAll()
  } catch (error) {
    console.error('创建失败:', error)
    message.error('创建失败')
  } finally {
    creating.value = false
  }
}

const handleTaskComplete = (task) => {
  if (task.name.includes('重建')) {
    refreshAll()
  }
}

const getTypeColor = (type) => {
  const colors = { md: 'blue', image: 'green', pdf: 'orange' }
  return colors[type] || 'default'
}

const getTypeName = (type) => {
  const names = { md: 'Markdown', image: '图片', pdf: 'PDF' }
  return names[type] || type || '未知'
}

const formatDate = (dateString) => {
  if (!dateString) return '未知'
  return dayjs(dateString).format('YYYY-MM-DD HH:mm')
}

// 生命周期
onMounted(() => {
  refreshAll()
})
</script>

<style scoped>
.data-manager {
  max-width: 1400px;
  margin: 0 auto;
}

.tree-node {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.node-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.node-actions {
  opacity: 0;
  transition: opacity 0.3s;
}

.tree-node:hover .node-actions {
  opacity: 1;
}

:deep(.ant-tree-node-content-wrapper) {
  width: 100%;
}

:deep(.ant-tree-title) {
  width: 100%;
}
</style>
