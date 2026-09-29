/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ratify: '#10b981',
        revise: '#3b82f6',
        refute: '#f59e0b',
        reject: '#ef4444',
      },
    },
  },
  plugins: [],
};
