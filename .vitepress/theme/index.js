import DefaultTheme from 'vitepress/theme'
import PytorchLink from './components/PytorchLink.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('PytorchLink', PytorchLink)
  }
}
