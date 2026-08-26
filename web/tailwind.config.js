/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'harness': {
          'bg': '#0a0a0f',
          'surface': '#12121a',
          'border': '#1e1e2e',
          'accent': '#6366f1',
          'success': '#22c55e',
          'warning': '#f59e0b',
          'danger': '#ef4444',
        }
      }
    },
  },
  plugins: [],
}
