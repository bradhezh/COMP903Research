import {defineConfig} from 'vite'
// for React transpilation and optimisation
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {alias: {'@': path.resolve(__dirname, 'src')}},
})
