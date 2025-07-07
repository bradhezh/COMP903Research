import {defineConfig} from 'vitest/config'
// for React transpilation and optimisation
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {alias: {'@': path.resolve(__dirname, 'src')}},
  test: {
    environment: 'jsdom',
    setupFiles: 'src/setupTests.ts',
    // making test funcs such as `test`, `describe`, and `expect` global for js
    globals: true,
  },
})
