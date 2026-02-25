import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  publicDir: 'public',
  server: {
    port: 3334,
    host: '127.0.0.1'
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 1000
  },
  assetsInclude: ['**/*.obj', '**/*.glb', '**/*.gltf', '**/*.hdr']
});
