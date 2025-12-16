import { useEffect, useRef } from "react";

interface Props {
  onFrame: (blob: Blob) => void;
}

function WebcamCapture({ onFrame }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    });

    const interval = setInterval(async () => {
      const video = videoRef.current;
      if (!video) return;

      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.drawImage(video, 0, 0);

      canvas.toBlob((blob) => {
        if (blob) onFrame(blob);
      }, "image/jpeg");
    }, 300);

    return () => clearInterval(interval);
  }, []);

  return (
    <video
      ref={videoRef}
      autoPlay
      playsInline
      className="w-[640px] h-[480px] object-cover rounded-lg"
    />
  );
}

export default WebcamCapture;
