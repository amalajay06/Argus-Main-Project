/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        argus: {
          black: "#0A0A0A",
          panel: "#131313",
          border: "#232323",
          accent: "#C2BBF0",
          "accent-dim": "#8B84B0",
          muted: "#8A8A8A",
        },
      },
      fontFamily: {
        display: ["Italiana", "serif"],
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
