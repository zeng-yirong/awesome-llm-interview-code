import React from "react";
import ReactDOM from "react-dom/client";
// katex-swap 与 katex.min.css 规则完全相同，只是每一条 @font-face 都带 font-display: swap
// —— 字体没加载完时先显示回退字体，而不是让公式整段不可见（FOIT）
import "katex/dist/katex-swap.min.css";
import "./index.css";
import App from "./App.tsx";

ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
