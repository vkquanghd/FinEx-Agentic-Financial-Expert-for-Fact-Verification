import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server 5173 (đã nằm trong CORS_ORIGINS của backend bootstrap.py).
// Muốn HTTPS dev (khớp backend --ssl): bật khối server.https bên dưới
// trỏ tới cùng cert self-signed của backend.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // https: {
    //   key: "../backend/certs/dev-key.pem",
    //   cert: "../backend/certs/dev-cert.pem",
    // },
  },
});
