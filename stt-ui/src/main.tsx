import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import WidgetView from "./components/WidgetView";
import OverlayView from "./overlay/OverlayView";
import "./styles/globals.css";

const windowType = new URLSearchParams(window.location.search).get("window");

const isTransparent = windowType === "widget" || windowType === "overlay";

if (isTransparent) {
  document.documentElement.style.background = "transparent";
  document.body.style.background = "transparent";
  document.getElementById("root")!.style.background = "transparent";
}

function Root() {
  switch (windowType) {
    case "widget":
      return <WidgetView />;
    case "overlay":
      return <OverlayView />;
    default:
      return <App />;
  }
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
);
