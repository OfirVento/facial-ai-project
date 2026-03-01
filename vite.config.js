import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  root: '.',
  publicDir: 'public',
  server: {
    port: 3334,
    host: '127.0.0.1'
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        lab: resolve(__dirname, 'lab.html'),
      },
    },
  },
  assetsInclude: ['**/*.obj', '**/*.glb', '**/*.gltf', '**/*.hdr']
});
