interface Props {
  boxes: any[];
}

function OverlayCanvas({ boxes }: Props) {
  return (
    <div className="absolute top-0 left-0 w-[640px] h-[480px] pointer-events-none">
      {boxes.map((box, idx) => {
        const [x1, y1, x2, y2] = box.box;
        return (
          <div
            key={idx}
            className="absolute border-2 border-red-500"
            style={{
              left: x1,
              top: y1,
              width: x2 - x1,
              height: y2 - y1,
            }}
          >
            <p className="text-red-400 bg-black/50 text-xs px-1">
              {box.cls} {box.score.toFixed(2)}
            </p>
          </div>
        );
      })}
    </div>
  );
}

export default OverlayCanvas;
