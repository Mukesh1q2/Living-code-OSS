import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // Base path for GitHub Pages deployment under project repo
  // Repo: Mukesh1q2/Living-code-OSS
  base: '/Living-code-OSS/',
  server: {
    port: 3000,
    host: true,
    hmr: {
      port: 3001
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
})
