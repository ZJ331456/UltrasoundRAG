<template>
  <div class="search-page">
    <!-- 页面标题 -->
    <a-page-header
      title="智能检索"
      sub-title="多模态检索系统"
      style="padding: 0 0 24px 0;"
    />

    <!-- 搜索表单 -->
    <a-card class="search-form" :bordered="false">
      <a-form layout="vertical">
        <!-- 检索模式选择 -->
        <a-form-item label="检索模式">
          <a-radio-group v-model:value="searchMode" button-style="solid">
            <a-radio-button value="t2t">文本→文本</a-radio-button>
            <a-radio-button value="t2i">文本→图片</a-radio-button>
            <a-radio-button value="i2t">图片→文本</a-radio-button>
            <a-radio-button value="i2i">图片→图片</a-radio-button>
            <a-radio-button value="auto">自动模式</a-radio-button>
            <a-radio-button value="multimodal">多模态</a-radio-button>
          </a-radio-group>
        </a-form-item>

        <!-- 文本查询 -->
        <a-form-item 
          v-if="needsTextInput" 
          label="查询文本"
        >
          <a-input
            v-model:value="queryText"
            placeholder="请输入查询内容，例如：心脏超声诊断"
            size="large"
            :maxlength="500"
            show-count
          />
        </a-form-item>

        <!-- 图片上传 -->
        <a-form-item 
          v-if="needsImageInput" 
          label="上传图片"
        >
          <a-upload-dragger
            v-model:file-list="fileList"
            :before-upload="handleBeforeUpload"
            @remove="handleRemoveFile"
            accept="image/*"
            :max-count="1"
          >
            <p class="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p class="ant-upload-text">点击或拖拽图片到此区域上传</p>
            <p class="ant-upload-hint">
              支持 JPG、PNG、JPEG 格式，文件大小不超过 10MB
            </p>
          </a-upload-dragger>
        </a-form-item>

        <!-- 高级选项 -->
        <a-form-item>
          <a-collapse ghost>
            <a-collapse-panel key="advanced" header="高级选项">
              <a-row :gutter="16">
                <a-col :span="12">
                  <a-form-item>
                    <template #label>
                      <a-space>
                        <span>集合选择</span>
                        <a-tooltip title="可以选择多个集合进行联合检索，提高搜索覆盖范围">
                          <QuestionCircleOutlined style="color: #1890ff;" />
                        </a-tooltip>
                      </a-space>
                    </template>
                    <a-select 
                      v-model:value="selectedCollections"
                      mode="multiple"
                      placeholder="选择要检索的集合（可多选）"
                      :max-tag-count="3"
                      show-search
                      :filter-option="filterCollectionOption"
                      allow-clear
                    >
                      <a-select-option 
                        v-for="collection in collections" 
                        :key="collection.name" 
                        :value="collection.name"
                      >
                        <a-space>
                          <span>{{ collection.name }}</span>
                          <a-tag size="small" color="blue">
                            {{ collection.document_count || 0 }}个文档
                          </a-tag>
                        </a-space>
                      </a-select-option>
                    </a-select>
                    <div style="margin-top: 4px; font-size: 12px; color: #666;">
                      已选择 {{ selectedCollections.length }} 个集合
                      <a-button 
                        v-if="collections.length > 0" 
                        type="link" 
                        size="small" 
                        @click="selectAllCollections"
                        style="padding: 0; height: auto; font-size: 12px;"
                      >
                        全选
                      </a-button>
                      <br>
                      <span v-if="totalSelectedDocuments === 0" style="color: #ff7875;">
                        ⚠️ 所选集合中共有 {{ totalSelectedDocuments }} 个文档
                      </span>
                      <span v-else style="color: #52c41a;">
                        ✅ 所选集合中共有 {{ totalSelectedDocuments }} 个文档
                      </span>
                    </div>
                  </a-form-item>
                </a-col>
                <a-col :span="6">
                  <a-form-item label="返回结果数">
                    <a-input-number 
                      v-model:value="topK"
                      :min="1"
                      :max="50"
                      style="width: 100%"
                    />
                  </a-form-item>
                </a-col>
                <a-col :span="6">
                  <a-form-item label="启用缓存">
                    <a-switch v-model:checked="enableCache" />
                  </a-form-item>
                </a-col>
              </a-row>
            </a-collapse-panel>
          </a-collapse>
        </a-form-item>

        <!-- 搜索按钮 -->
        <a-form-item>
          <a-button 
            type="primary" 
            size="large" 
            block
            :loading="searching"
            @click="performSearch"
            :disabled="!canSearch"
          >
            <SearchOutlined />
            开始检索
          </a-button>
        </a-form-item>
      </a-form>
    </a-card>

    <!-- 搜索结果 -->
    <a-card 
      v-if="searchResults.length > 0 || hasSearched"
      title="检索结果" 
      class="search-results"
      :bordered="false"
      style="margin-top: 24px;"
    >
      <template #extra>
        <a-space>
          <span>找到 {{ searchResults.length }} 个结果</span>
          <span>耗时 {{ searchTime }}ms</span>
        </a-space>
      </template>

      <div v-if="searchResults.length === 0 && hasSearched" class="no-results">
        <a-empty description="未找到相关结果">
          <template #description>
            <div>
              <p>未找到相关结果</p>
              <p v-if="totalSelectedDocuments === 0" style="color: #ff7875; font-size: 14px;">
                所选集合中没有文档，请先添加数据
              </p>
            </div>
          </template>
          <a-space>
            <a-button type="primary" @click="clearResults">
              重新搜索
            </a-button>
            <a-button v-if="totalSelectedDocuments === 0" @click="$router.push('/app/collection')">
              管理数据
            </a-button>
          </a-space>
        </a-empty>
      </div>

      <div v-else>
        <a-list
          :data-source="searchResults"
          :pagination="{
            pageSize: 10,
            showTotal: (total) => `共 ${total} 条结果`,
            showSizeChanger: false
          }"
        >
          <template #renderItem="{ item, index }">
            <a-list-item>
              <a-list-item-meta>
                <template #title>
                  <a-space>
                    <span>结果 {{ index + 1 }}</span>
                    <a-tag 
                      :color="getScoreColor(item.score)"
                    >
                      得分: {{ item.score?.toFixed(4) || 'N/A' }}
                    </a-tag>
                    <a-tag v-if="item.retrieval_type" color="blue">
                      {{ item.retrieval_type }}
                    </a-tag>
                  </a-space>
                </template>
                <template #description>
                  <!-- 文档内容 -->
                  <div class="result-content" style="margin-bottom: 12px; line-height: 1.6;">
                    <div v-if="item.content" style="max-height: 200px; overflow-y: auto; padding: 8px; background: #fafafa; border-radius: 4px;">
                      {{ item.content }}
                    </div>
                    <div v-else style="color: #999; font-style: italic;">
                      无内容描述
                    </div>
                  </div>
                  
                  <!-- 图片显示 -->
                  <div v-if="item.image_url || item.image_path" style="margin-bottom: 12px;">
                    <img 
                      :src="getImageUrl(item)" 
                      :alt="`检索结果图片 ${index + 1}`"
                      style="max-width: 200px; max-height: 150px; border-radius: 4px; border: 1px solid #d9d9d9;"
                      @error="handleImageError"
                    />
                  </div>
                  
                  <!-- 元数据 -->
                  <div style="margin-bottom: 8px;">
                    <a-space wrap>
                      <a-tag v-if="item.resource_collection" color="green">
                        集合: {{ item.resource_collection }}
                      </a-tag>
                      <a-tag v-if="item.doc_id" color="orange">
                        ID: {{ item.doc_id }}
                      </a-tag>
                      <a-tag v-if="item.metadata?.file_type" color="purple">
                        类型: {{ item.metadata.file_type }}
                      </a-tag>
                    </a-space>
                  </div>
                  
                  <!-- 额外元数据 -->
                  <div v-if="item.metadata && Object.keys(item.metadata).length > 0">
                    <a-collapse ghost size="small">
                      <a-collapse-panel key="metadata" header="查看详细信息">
                        <pre style="background: #f5f5f5; padding: 8px; border-radius: 4px; font-size: 12px; max-height: 150px; overflow-y: auto;">{{ JSON.stringify(item.metadata, null, 2) }}</pre>
                      </a-collapse-panel>
                    </a-collapse>
                  </div>
                </template>
              </a-list-item-meta>
            </a-list-item>
          </template>
        </a-list>
      </div>
    </a-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import { SearchOutlined, InboxOutlined, QuestionCircleOutlined } from '@ant-design/icons-vue'
import { searchAPI } from '../api/index.js'
import { useSystemStore } from '../stores/system'
import { useCollectionStore } from '../stores/collection'

// 状态管理
const systemStore = useSystemStore()
const collectionStore = useCollectionStore()

// 响应式数据
const searchMode = ref('t2t')
const queryText = ref('')
const fileList = ref([])
const selectedCollections = ref([])
const topK = ref(10)
const enableCache = ref(true)
const searching = ref(false)
const searchResults = ref([])
const searchTime = ref(0)
const hasSearched = ref(false)

// 计算属性
const databases = computed(() => systemStore.databases)
const collections = computed(() => collectionStore.collections)

const totalSelectedDocuments = computed(() => {
  return selectedCollections.value.reduce((total, collectionName) => {
    const collection = collections.value.find(c => c.name === collectionName)
    return total + (collection?.document_count || 0)
  }, 0)
})

const needsTextInput = computed(() => {
  return ['t2t', 't2i', 'auto', 'multimodal'].includes(searchMode.value)
})

const needsImageInput = computed(() => {
  return ['i2t', 'i2i', 'auto', 'multimodal'].includes(searchMode.value)
})

const canSearch = computed(() => {
  // 检查是否选择了集合
  if (selectedCollections.value.length === 0) {
    return false
  }
  
  // 检查选中的集合是否有文档（允许搜索空集合，但会给出警告）
  // if (totalSelectedDocuments.value === 0) {
  //   return false
  // }
  
  if (needsTextInput.value && !queryText.value.trim()) {
    return false
  }
  if (needsImageInput.value && fileList.value.length === 0) {
    if (searchMode.value === 'auto' || searchMode.value === 'multimodal') {
      // 自动模式和多模态模式可以只有文本或只有图片
      return queryText.value.trim() || fileList.value.length > 0
    }
    return false
  }
  return true
})

// 方法
const filterCollectionOption = (input, option) => {
  return option.children().toLowerCase().includes(input.toLowerCase())
}

const selectAllCollections = () => {
  selectedCollections.value = collections.value.map(c => c.name)
}

const handleBeforeUpload = (file) => {
  // 检查文件类型
  const isImage = file.type.startsWith('image/')
  if (!isImage) {
    message.error('只能上传图片文件!')
    return false
  }

  // 检查文件大小
  const isLt10M = file.size / 1024 / 1024 < 10
  if (!isLt10M) {
    message.error('图片大小不能超过 10MB!')
    return false
  }

  return false // 阻止自动上传
}

const handleRemoveFile = (file) => {
  fileList.value = fileList.value.filter(item => item.uid !== file.uid)
  return true
}

const performSearch = async () => {
  try {
    searching.value = true
    hasSearched.value = true
    const startTime = Date.now()

    // 检查是否选择了集合
    if (selectedCollections.value.length === 0) {
      message.warning('请至少选择一个集合进行检索')
      return
    }
    
    // 检查选中集合是否有文档
    if (totalSelectedDocuments.value === 0) {
      message.warning('所选集合中没有文档，检索可能无结果。请确认集合中是否已添加数据。')
      // 仍然允许搜索，但给出警告
    }

    let response
    
    // 按类型分类集合
    const textCollections = []
    const imageCollections = []
    
    selectedCollections.value.forEach(collectionName => {
      const collection = collections.value.find(c => c.name === collectionName)
      if (collection) {
        // 根据集合类型或名称判断是文本还是图片集合
        if (collection.type === 'image' || collectionName.includes('image')) {
          imageCollections.push(collectionName)
        } else {
          textCollections.push(collectionName)
        }
      } else {
        // 如果找不到集合信息，根据名称推断
        if (collectionName.includes('image')) {
          imageCollections.push(collectionName)
        } else {
          textCollections.push(collectionName)
        }
      }
    })

    // 如果有图片上传，使用基于集合的文件上传接口
    if (needsImageInput.value && fileList.value.length > 0) {
      const formData = new FormData()
      formData.append('file', fileList.value[0].originFileObj)
      formData.append('mode', searchMode.value)
      formData.append('top_k', topK.value.toString())
      
      // 添加文本和图片集合
      textCollections.forEach(collection => {
        formData.append('text_collections', collection)
      })
      imageCollections.forEach(collection => {
        formData.append('image_collections', collection)
      })
      
      if (queryText.value.trim()) {
        formData.append('query', queryText.value.trim())
      }

      response = await searchAPI.collectionUploadSearch(formData)
    } else {
      // 使用基于集合的搜索接口
      const searchParams = {
        mode: searchMode.value,
        text_collections: textCollections,
        image_collections: imageCollections,
        top_k: topK.value,
        query: queryText.value.trim() || null
      }

      response = await searchAPI.collectionSearch(searchParams)
    }

    searchTime.value = Date.now() - startTime

    if (response.data.success) {
      const results = response.data.data?.results || []
      const dataError = response.data.data?.error
      
      searchResults.value = results
      
      // 检查是否有数据层面的错误
      if (dataError) {
        if (results.length === 0) {
          message.warning(`检索完成但无结果：${dataError}`)
        } else {
          message.info(`检索完成，找到 ${results.length} 个结果（注意：${dataError}）`)
        }
      } else {
        if (results.length === 0) {
          message.info('检索完成，但未找到相关结果')
        } else {
          message.success(`检索完成，找到 ${results.length} 个结果`)
        }
      }
    } else {
      throw new Error(response.data.error || '检索失败')
    }

  } catch (error) {
    console.error('检索失败:', error)
    searchResults.value = []
  } finally {
    searching.value = false
  }
}

const clearResults = () => {
  searchResults.value = []
  hasSearched.value = false
  queryText.value = ''
  fileList.value = []
}

const getScoreColor = (score) => {
  if (!score) return 'default'
  if (score > 0.8) return 'green'
  if (score > 0.6) return 'blue'
  if (score > 0.4) return 'orange'
  return 'red'
}

const getImageUrl = (item) => {
  // 如果有完整的URL，直接使用
  if (item.image_url) {
    return item.image_url
  }
  
  // 如果是相对路径，构建完整URL
  if (item.image_path) {
    // 检查是否是绝对路径
    if (item.image_path.startsWith('http')) {
      return item.image_path
    }
    // 构建相对于API服务器的URL
    return `${window.location.origin}/static/images/${item.image_path}`
  }
  
  // 检查metadata中的图片信息
  if (item.metadata?.image_path) {
    return `${window.location.origin}/static/images/${item.metadata.image_path}`
  }
  
  // 检查其他可能的图片字段
  if (item.metadata?.image_url) {
    return item.metadata.image_url
  }
  
  return ''
}

const handleImageError = (event) => {
  // 图片加载失败时的处理
  console.warn('图片加载失败:', event.target.src)
  event.target.style.display = 'none'
}

// 生命周期
onMounted(async () => {
  // 加载数据库和集合列表
  await Promise.all([
    systemStore.loadDatabases(),
    collectionStore.loadCollections()
  ])
  
  // 默认选择前3个集合（如果有的话）
  if (collections.value.length > 0) {
    selectedCollections.value = collections.value
      .slice(0, 3)
      .map(c => c.name)
  }
})
</script>

<style scoped>
.search-page {
  max-width: 1200px;
  margin: 0 auto;
}

.search-form {
  margin-bottom: 24px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.search-results {
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.no-results {
  text-align: center;
  padding: 48px 24px;
}

:deep(.ant-upload-drag) {
  background-color: #fafafa;
  border: 2px dashed #d9d9d9;
  border-radius: 8px;
  padding: 40px;
  transition: all 0.3s ease;
}

:deep(.ant-upload-drag:hover) {
  border-color: #1890ff;
  background-color: #f0f9ff;
}

:deep(.ant-list-item-meta-description) {
  color: rgba(0, 0, 0, 0.65);
  font-size: 14px;
  line-height: 1.5;
}
</style>
