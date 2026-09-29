/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        darkBg: '#0f172a',      // slate-900
        darkCard: '#1e293b',    // slate-800
        darkBorder: '#334155',  // slate-700
        accentCyan: '#22d3ee',  // cyan-400
        warnAmber: '#fbbf24',   // amber-400
        dangerRed: '#ef4444',   // red-500
        successGreen: '#22c55e',// green-500
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', 'ui-monospace', 'monospace'],
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
