import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

import path from 'node:path'
// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // Read env files from the monorepo root (../.env, ../.env.local, etc.)
  envDir: path.resolve(__dirname, '..'),
  server: {
    port: 5173,
    strictPort: true,
    host: '127.0.0.1',
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/setupTests.ts',
    globals: true,
    css: true,
  },
})
