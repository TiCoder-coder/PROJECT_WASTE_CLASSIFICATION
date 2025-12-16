import { useState } from "react";
import WebcamCapture from "./WebcamCapture";
import OverlayCanvas from "./OverlayCanvas";
import { detectFrame } from "./api";

function App() {
  const [boxes, setBoxes] = useState<any[]>([]);

  const handleFrame = async (blob: Blob) => {
    try {
      const res = await detectFrame(blob);
      setBoxes(res.detections || []);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="w-screen h-screen flex items-center justify-center">
      <div className="relative">
        <WebcamCapture onFrame={handleFrame} />
        <OverlayCanvas boxes={boxes} />
      </div>
    </div>
  );
}

export default App;
