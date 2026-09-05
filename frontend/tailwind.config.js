/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#14161C",
        "ink-raised": "#1B1E26",
        paper: "#F6F5F1",
        "paper-dim": "#EBE9E2",
        teal: {
          DEFAULT: "#1F7A6C",
          bright: "#2F9C8A",
        },
        status: {
          high: "#3FA672",
          medium: "#D9A441",
          low: "#C0503F",
          neutral: "#7D8494",
        },
      },
      fontFamily: {
        serif: ["\"Source Serif 4\"", "Georgia", "serif"],
        sans: ["\"IBM Plex Sans\"", "system-ui", "-apple-system", "sans-serif"],
      },
      maxWidth: {
        memo: "700px",
      },
    },
  },
  plugins: [],
};
