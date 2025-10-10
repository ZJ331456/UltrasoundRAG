import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { createRouter, createWebHistory } from 'vue-router'
import Antd from 'ant-design-vue'
import 'ant-design-vue/dist/reset.css'
import './style.css'

import App from './App.vue'

// 导入页面组件
import Dashboard from './views/Dashboard.vue'
import Search from './views/Search.vue'
import SystemStatus from './views/SystemStatus.vue'
import DataManager from './views/DataManager.vue'

// 路由配置
const routes = [
  { path: '/app', redirect: '/app/dashboard' },
  { path: '/app/dashboard', name: 'Dashboard', component: Dashboard },
  { path: '/app/search', name: 'Search', component: Search },
  { path: '/app/data', name: 'DataManager', component: DataManager },
  { path: '/app/status', name: 'SystemStatus', component: SystemStatus },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

// Pinia状态管理
const pinia = createPinia()

// 创建应用
const app = createApp(App)

app.use(pinia)
app.use(router)
app.use(Antd)

app.mount('#app')
